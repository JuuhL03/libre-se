"""
Servidor da versao web do Libre-se (FastAPI).

  GET  /                 a pagina (pasta web/)
  GET  /api/info         modos, deteccao de cada modo e palavras do modelo
  GET  /modelos/<arq>    modelos do MediaPipe para o navegador (maos e corpo)
  WS   /ws               uma sessao por conexao: o navegador manda os pontos de
                         cada quadro e recebe o estado do painel

Rodar no computador (para testar):
  pip install -r servidor/requirements.txt
  uvicorn servidor.app:app --port 8000      -> abra http://localhost:8000

Variaveis de ambiente:
  LIBRES_DADOS    pasta da calibracao de cada usuario (no Railway: o volume, ex. /data)
  LIBRES_CODIGO   se definida, a pagina pede este codigo para entrar
  (pagina com ?pesquisa=1: modo pesquisa do teste com usuarios, anonimo e sem salvar nada)
  LIBRES_SESSOES  maximo de pessoas usando ao mesmo tempo (padrao 12)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from servidor.motor import RAIZ, Recursos, Sessao

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")

WEB = RAIZ / "web"
MODELOS = {   # nome no navegador -> onde procurar (a pasta do projeto primeiro)
    "hand_landmarker.task": [RAIZ / "hand_landmarker.task", RAIZ / "servidor" / "modelos" / "hand_landmarker.task"],
    "pose_landmarker_lite.task": [RAIZ / "pose_landmarker_lite.task",
                                  RAIZ / "servidor" / "modelos" / "pose_landmarker_lite.task"],
}
CODIGO = os.environ.get("LIBRES_CODIGO", "").strip()
MAX_SESSOES = int(os.environ.get("LIBRES_SESSOES", "12"))
MAX_QUADROS_POR_MSG = 90          # (o tamanho maximo de cada mensagem e o --ws-max-size do uvicorn)

TEMPO_SESSAO = 20 * 60             # s: a sessao (texto, modo, calibracao em andamento) espera a
                                   # pagina reconectar depois de uma queda da internet
MAX_GUARDADAS = 200

recursos = Recursos()
sessoes_ativas = 0
guardadas: dict[str, list] = {}    # sid -> [Sessao, trava, ultimo uso]


@asynccontextmanager
async def ciclo(_app):
    if sys.platform == "win32":
        # no Windows, recarregar a pagina fecha a conexao "a forca" e o Python imprime um
        # ConnectionResetError [WinError 10054] assustador, mas inofensivo: silencia so ele
        loop = asyncio.get_running_loop()
        padrao = loop.get_exception_handler()

        def tratador(lp, contexto):
            if isinstance(contexto.get("exception"), ConnectionResetError):
                return
            if padrao:
                padrao(lp, contexto)
            else:
                lp.default_exception_handler(contexto)
        loop.set_exception_handler(tratador)
    yield


app = FastAPI(title="Libre-se web", docs_url=None, redoc_url=None, lifespan=ciclo)


def _soltar(sid: str) -> None:
    g = guardadas.pop(sid, None)
    if g is not None:
        g[0].descartar()               # modo pesquisa: apaga os exemplos temporarios


async def obter_sessao(sid: str, usuario: str, modo: str | None, pesquisa: bool = False) -> list:
    """A sessao desta pagina: a mesma de antes se ela so caiu e reconectou."""
    agora = time.time()
    for k, (_s, _tr, t) in list(guardadas.items()):
        if agora - t > TEMPO_SESSAO:
            _soltar(k)
    if pesquisa:                       # anonimo: o nome vem do id aleatorio da pagina
        usuario = "pesquisa-" + "".join(c for c in sid if c.isalnum())[:10]
    g = guardadas.get(sid) if sid else None
    if g is None or g[0].usuario != (usuario or "padrao").strip()[:40] or g[0].efemera != pesquisa:
        if g is not None:
            _soltar(sid)
        s = await asyncio.to_thread(Sessao, recursos, usuario, 5, modo, pesquisa)
        g = [s, threading.Lock(), agora]
        if sid:
            guardadas[sid] = g
            if len(guardadas) > MAX_GUARDADAS:
                _soltar(min(guardadas, key=lambda k: guardadas[k][2]))
    g[2] = agora
    return g


def _travado(g: list, funcao, *args):
    with g[1]:
        return funcao(*args)


def _modelo(nome: str) -> Path | None:
    for c in MODELOS.get(nome, []):
        if c.exists() and c.stat().st_size > 100_000:
            return c
    return None


@app.get("/saude")
def saude() -> dict:
    return {"ok": True, "modos": recursos.modos(), "sessoes": sessoes_ativas,
            "modelos_navegador": {n: _modelo(n) is not None for n in MODELOS}}


@app.get("/api/info")
def info() -> JSONResponse:
    d = recursos.info()
    d["precisa_codigo"] = bool(CODIGO)
    d["modelos_navegador"] = {n: _modelo(n) is not None for n in MODELOS}
    return JSONResponse(d, headers={"Cache-Control": "no-store"})


@app.get("/modelos/{nome}")
def modelo(nome: str) -> Response:
    caminho = _modelo(nome)
    if caminho is None:
        return JSONResponse({"erro": f"modelo {nome} nao encontrado no servidor"}, status_code=404)
    return FileResponse(caminho, media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


def _codigo_ok(recebido: str) -> bool:
    return not CODIGO or hmac.compare_digest(recebido.strip().encode(), CODIGO.encode())


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    global sessoes_ativas
    await websocket.accept()
    qp = websocket.query_params
    if not _codigo_ok(qp.get("codigo", "")):
        await websocket.send_json({"tipo": "erro", "erro": "codigo", "msg": "codigo de acesso errado"})
        await websocket.close(code=4001)
        return
    if sessoes_ativas >= MAX_SESSOES:
        await websocket.send_json({"tipo": "erro", "erro": "cheio", "msg": "servidor cheio: tente daqui a pouco"})
        await websocket.close(code=4002)
        return
    sessoes_ativas += 1
    try:
        try:
            g = await obter_sessao(qp.get("sid", "")[:64], qp.get("usuario", "padrao"), qp.get("modo"),
                                   qp.get("pesquisa") == "1")
        except Exception as e:  # noqa: BLE001
            await websocket.send_json({"tipo": "erro", "erro": "sessao", "msg": f"nao abriu a sessao: {e}"})
            await websocket.close(code=4003)
            return
        sessao = g[0]
        estado = await asyncio.to_thread(_travado, g, sessao.estado)
        await websocket.send_json({"tipo": "config", "usuario": sessao.usuario, **recursos.info(),
                                   "estado": estado})
        while True:
            texto = await websocket.receive_text()
            try:
                msg = json.loads(texto)
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            tipo = msg.get("tipo")
            g[2] = time.time()
            if tipo == "quadros":
                q = msg.get("q") or []
                if not isinstance(q, list):
                    q = []
                await asyncio.to_thread(_travado, g, sessao.quadros, q[-MAX_QUADROS_POR_MSG:])
            elif tipo == "cmd":
                cmd = str(msg.get("cmd", ""))[:20]
                await asyncio.to_thread(_travado, g, sessao.comando, cmd, msg)
            elif tipo == "ping":
                await websocket.send_json({"tipo": "pong", "t": msg.get("t"), "servidor": time.time()})
                continue
            else:
                continue
            estado, extras = await asyncio.to_thread(_travado, g, lambda: (sessao.estado(), sessao.pegar_saidas()))
            for extra in extras:                       # antes do estado: quem espera o "ack" ja recebeu tudo
                await websocket.send_json(extra)
            await websocket.send_json({"tipo": "estado", "ack": msg.get("id"), **estado})
    except WebSocketDisconnect:
        pass
    finally:
        sessoes_ativas -= 1
        if not qp.get("sid") and "g" in locals():
            g[0].descartar()           # sessao sem id nao fica guardada: apaga ja


# a pagina (por ultimo: as rotas acima tem prioridade)
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

#!/usr/bin/env python3
"""
Teste da versao web sem camera (1-2 minutos): sobe o servidor nesta maquina,
conecta como a pagina faz (WebSocket, lotes de quadros) e usa sinais
sinteticos de uma pessoa (os mesmos do teste_calibracao.py):

  1. CALIBRADO: troca o vocabulario, calibra seguindo o painel (a palavra da
     vez), reconhece 4 sinais e recusa um sinal nao calibrado;
  2. a conexao cai e volta: o texto continua;
  3. ALFABETO e PALAVRAS respondem (se os modelos existirem);
  4. quem ainda nao sabe os sinais: com um perfil "referencia" gravado, a
     tecla V mostra como a referencia faz (e alterna com o proprio exemplo).

Uso (na pasta do projeto):  python servidor/teste_servidor.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
DADOS = Path(tempfile.mkdtemp(prefix="libres_teste_web_"))
os.environ["LIBRES_DADOS"] = str(DADOS)
os.environ.pop("LIBRES_CODIGO", None)

import uvicorn  # noqa: E402
import websockets  # noqa: E402

import teste_calibracao as TC  # noqa: E402
import teste_v3 as TV  # noqa: E402
from servidor.app import app  # noqa: E402

FPS = 30
VOCAB = ["oi", "bom dia", "2", "tchau"]
falhas: list[str] = []


def confere(ok: bool, msg: str) -> None:
    print(("  OK    " if ok else "  FALHA ") + msg)
    if not ok:
        falhas.append(msg)


def porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cliente:
    def __init__(self, ws) -> None:
        self.ws, self.id, self.estado, self.exemplos = ws, 0, None, []
        self.t = time.time()

    def q(self, maos, lado, pose) -> dict:
        self.t += 1 / FPS
        r = lambda v: None if not np.isfinite(v) else round(float(v), 4)  # noqa: E731
        m = [None if np.isnan(maos[k, 0, 0]) else [r(v) for v in maos[k].reshape(-1)] for k in range(2)]
        p = None if pose is None else [r(v) for v in np.asarray(pose).reshape(-1)]
        return {"t": round(self.t, 6), "a": round(16 / 9, 4), "m": m, "l": [int(x) for x in lado], "p": p}

    async def _ate_ack(self, ident) -> dict:
        while True:
            msg = json.loads(await self.ws.recv())
            if msg["tipo"] == "exemplo":
                self.exemplos.append(msg)
            elif msg["tipo"] == "estado":
                self.estado = msg
                if msg.get("ack") == ident:
                    return msg

    async def quadros(self, lista) -> dict:
        for i in range(0, len(lista), 3):
            self.id += 1
            await self.ws.send(json.dumps({"tipo": "quadros", "id": self.id, "q": lista[i:i + 3]}))
            await self._ate_ack(self.id)
        return self.estado

    async def cmd(self, cmd, **extra) -> dict:
        self.id += 1
        await self.ws.send(json.dumps({"tipo": "cmd", "cmd": cmd, "id": self.id, **extra}))
        return await self._ate_ack(self.id)


async def principal(url: str) -> None:
    rng = np.random.default_rng(11)
    pessoa = TV.Pessoa(41, rng)
    pessoa.desvio = {p: rng.uniform(-0.04, 0.04, 2) for p in TC.SINAIS}
    vazio = (np.full((2, 21, 3), np.nan), np.full(2, -1), None)

    def sinal(c: Cliente, palavra: str) -> list:
        return [c.q(m, l, p) for m, l, p in TC.fluxo(pessoa, palavra, rng, FPS)] + [c.q(*vazio) for _ in range(9)]

    sid = f"teste-{time.time():.0f}"
    async with websockets.connect(f"{url}?{urlencode({'usuario': 'teste web', 'sid': sid, 'modo': 'CALIBRADO'})}") as ws:
        cfg = json.loads(await ws.recv())
        confere(cfg["tipo"] == "config" and "CALIBRADO" in cfg["modos"], f"conectou; modos {cfg['modos']}")
        c = Cliente(ws)
        if cfg["estado"]["cal"]["calibrando"]:
            await c.cmd("k")
        e = await c.cmd("vocabulario", palavras=VOCAB)
        confere([p for p, _ in e["cal"]["palavras"]] == VOCAB, "vocabulario trocado pela pagina")
        n = 0
        while c.estado["cal"]["calibrando"] and n < 60:
            await c.quadros(sinal(c, c.estado["cal"]["palavra"]))
            n += 1
        e = c.estado
        confere(not e["cal"]["calibrando"] and e["cal"]["tem_reconhecedor"],
                f"calibracao pronta com {n} sinais: {e['cal']['resumo']}")
        await c.cmd("v")
        confere(bool(c.exemplos) and len(c.exemplos[-1]["maos"]) > 5, "tecla V: exemplo gravado chegou")
        for palavra in ["tchau", "oi", "2", "bom dia"]:
            await c.quadros(sinal(c, palavra))
        confere(c.estado["texto"] == "TCHAU OI 2 BOM DIA", f"reconheceu: {c.estado['texto']!r}")
        await c.quadros(sinal(c, "pão"))
        confere(c.estado["ultimo"]["ok"] is None, "sinal nao calibrado (pao): nao reconhecido")
    async with websockets.connect(f"{url}?{urlencode({'usuario': 'teste web', 'sid': sid})}") as ws:
        cfg = json.loads(await ws.recv())
        confere(cfg["estado"]["texto"] == "TCHAU OI 2 BOM DIA", "a conexao caiu e voltou: o texto continua")
        c = Cliente(ws)
        for modo in ("ALFABETO", "PALAVRAS"):
            if modo in cfg["modos"]:
                e = await c.cmd("modo", alvo=modo)
                await c.quadros(sinal(c, "oi"))
                confere(c.estado["modo"] == modo, f"modo {modo} responde")
            else:
                print(f"  (modo {modo} sem modelo neste servidor: pulado)")

    # 4. alguem que conhece os sinais calibrou como "referencia" (aqui: a copia do perfil acima)
    cal = DADOS / "calibracao"
    shutil.copytree(cal / "teste_web", cal / "referencia")
    vocab = VOCAB + ["pão"]
    async with websockets.connect(f"{url}?{urlencode({'usuario': 'novato', 'sid': sid + '-novato', 'modo': 'CALIBRADO'})}") as ws:
        cfg = json.loads(await ws.recv())
        c = Cliente(ws)
        if cfg["estado"]["cal"]["calibrando"]:
            await c.cmd("k")
        e = await c.cmd("vocabulario", palavras=vocab)
        confere(e["cal"]["calibrando"] and e["cal"]["palavra"] == "oi" and e["cal"]["ref"] is True,
                "novato calibrando 'oi': o painel sabe que a referencia tem esse sinal")
        n0 = len(c.exemplos)
        await c.cmd("v")
        confere(len(c.exemplos) == n0 + 1 and c.exemplos[-1]["referencia"] and c.exemplos[-1]["palavra"] == "oi",
                "V sem exemplo proprio: mostra como a referencia faz")
        await c.quadros(sinal(c, "oi"))
        confere(c.estado["cal"]["palavra"] == "oi" and c.estado["cal"]["n"] == 1, "gravou o 1o exemplo de 'oi'")
        await c.cmd("v")
        confere(not c.exemplos[-1]["referencia"], "depois de gravar: V mostra o proprio exemplo")
        await c.cmd("v")
        confere(c.exemplos[-1]["referencia"], "V de novo enquanto passa: alterna para o da referencia")
        await c.cmd("v")
        confere(not c.exemplos[-1]["referencia"], "V de novo: volta para o proprio")
        for _ in range(len(vocab)):
            if c.estado["cal"]["palavra"] == "pão":
                break
            await c.cmd("n")
        confere(c.estado["cal"]["palavra"] == "pão" and c.estado["cal"]["ref"] is False,
                "'pão': a referencia nao gravou")
        n0 = len(c.exemplos)
        e = await c.cmd("v")
        confere(len(c.exemplos) == n0 and "dicionario" in (e["cal"]["msg"] or ""),
                f"V sem exemplo nenhum: manda para o dicionario ({e['cal']['msg']!r})")


def main() -> None:
    porta = porta_livre()
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=porta, log_level="warning"))
    threading.Thread(target=servidor.run, daemon=True).start()
    for _ in range(100):
        if servidor.started:
            break
        time.sleep(0.1)
    print(f"servidor de teste na porta {porta} (dados em {DADOS})")
    try:
        asyncio.run(principal(f"ws://127.0.0.1:{porta}/ws"))
    finally:
        servidor.should_exit = True
        shutil.rmtree(DADOS, ignore_errors=True)
    print("\n" + ("TUDO CERTO" if not falhas else f"{len(falhas)} FALHA(S): " + "; ".join(falhas)))
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()

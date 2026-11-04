// Endereco do servidor do Libre-se (libre-se/servidor/app.py), o mesmo da versao web.
//
// - Producao: o deploy do Railway.
// - Teste local: suba o servidor com
//     uvicorn servidor.app:app --host 0.0.0.0 --port 8000
//   descubra o IP do computador (ipconfig getifaddr en0) e troque SERVIDOR_URL
//   por 'http://<IP>:8000' (celular e computador na mesma rede).
export const SERVIDOR_URL = 'https://libre-se-web-production.up.railway.app';

// O mesmo endereco no protocolo do WebSocket (http -> ws, https -> wss).
export const SERVIDOR_WS = SERVIDOR_URL.replace(/^http/, 'ws');

// Codigo de acesso (LIBRES_CODIGO no Railway); vazio se o servidor nao pede.
export const CODIGO_ACESSO = '';

// Perfil padrao: a calibracao fica salva no servidor com este nome.
export const USUARIO_PADRAO = 'app';

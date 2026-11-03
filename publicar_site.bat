@echo off
rem Confere e publica a versao web do Libre-se no Railway.
rem Uso: dois cliques neste arquivo (na pasta do projeto, com o .venv criado).
rem Os registros ficam em publicar.log, publicar_teste.log e publicar_railway.log.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo [%date% %time%] inicio> publicar.log
if not exist ".venv\Scripts\activate.bat" (
  echo sem .venv nesta pasta>> publicar.log
  echo Nao achei o .venv nesta pasta. Crie com: python -m venv .venv
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo.
echo 1/2  Testando o servidor com os seus modelos (1 a 2 minutos)...
python servidor\teste_servidor.py > publicar_teste.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] teste do servidor FALHOU>> publicar.log
  echo O teste do servidor falhou: veja publicar_teste.log. Nada foi publicado.
  pause
  exit /b 1
)
echo [%date% %time%] teste do servidor OK>> publicar.log
echo 2/2  Publicando no Railway (alguns minutos)...
where railway > nul 2>&1
if errorlevel 1 (
  echo [%date% %time%] railway nao instalado>> publicar.log
  echo O Railway CLI nao esta instalado: npm i -g @railway/cli
  pause
  exit /b 1
)
railway up > publicar_railway.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] railway up FALHOU>> publicar.log
  echo A publicacao falhou: veja publicar_railway.log
  pause
  exit /b 1
)
echo [%date% %time%] railway up OK>> publicar.log
echo.
echo Pronto: o site foi publicado. Pode fechar esta janela.
pause

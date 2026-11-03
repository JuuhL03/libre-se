@echo off
rem Numeros do TCC (secao 3.7.3): reconhecimento de palavras com uma pessoa fora do treino (LOSO).
rem 1) MINDS-Libras: LOSO por sinalizante + divisao aleatoria 80/20 + comparacao com 1 mao (v2)
rem 2) LIBRAS-EQT-UECE: LOSO por informante
rem Demora 1 a 3 horas; pode usar o computador enquanto isso. Resultados: avaliacao_minds.log e avaliacao_eqt.log
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
call ".venv\Scripts\activate.bat"
echo [%date% %time%] inicio> avaliacao_minds.log
python avaliar_v3.py --dados minds --recortar --sem-mao-parada --aleatoria --comparar >> avaliacao_minds.log 2>&1
echo [%date% %time%] fim (codigo %errorlevel%)>> avaliacao_minds.log
echo [%date% %time%] inicio> avaliacao_eqt.log
python avaliar_v3.py --dados eqt --recortar --sem-mao-parada --aleatoria >> avaliacao_eqt.log 2>&1
echo [%date% %time%] fim (codigo %errorlevel%)>> avaliacao_eqt.log
echo.
echo Pronto. Resultados em avaliacao_minds.log e avaliacao_eqt.log
pause

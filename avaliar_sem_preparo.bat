@echo off
rem Confere as linhas "sem o preparo" das Tabelas 2 e 3 do TCC (versao com duas maos e corpo,
rem sem o recorte do repouso e sem ignorar a mao parada), com uma pessoa fora do treino (LOSO).
rem Demora cerca de 1 hora. Resultados: avaliacao_minds_sem_preparo.log e avaliacao_eqt_sem_preparo.log
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
call ".venv\Scripts\activate.bat"
echo [%date% %time%] inicio> avaliacao_minds_sem_preparo.log
python avaliar_v3.py --dados minds >> avaliacao_minds_sem_preparo.log 2>&1
echo [%date% %time%] fim (codigo %errorlevel%)>> avaliacao_minds_sem_preparo.log
echo [%date% %time%] inicio> avaliacao_eqt_sem_preparo.log
python avaliar_v3.py --dados eqt >> avaliacao_eqt_sem_preparo.log 2>&1
echo [%date% %time%] fim (codigo %errorlevel%)>> avaliacao_eqt_sem_preparo.log
echo.
echo Pronto. Resultados em avaliacao_minds_sem_preparo.log e avaliacao_eqt_sem_preparo.log
pause

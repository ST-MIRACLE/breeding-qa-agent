@echo off
chcp 65001 >nul 2>&1
cd /d "D:\ai工作流\breeding-qa-agent"
echo ========================================
echo   番茄育种知识库问答系统
echo ========================================
echo.
echo 正在初始化系统，请稍候...
python src/chat_server.py
pause

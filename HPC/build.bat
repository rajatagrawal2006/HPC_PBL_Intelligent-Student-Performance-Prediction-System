@echo off
g++ -std=c++17 -O2 -Wall -fopenmp -static -Ithird_party src\main.cpp src\pipeline.cpp -o sps.exe -lws2_32
if errorlevel 1 (echo Build failed & exit /b 1)
echo Build OK. Run sps.exe, then open http://127.0.0.1:8080


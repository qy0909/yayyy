@echo off
REM Complete Pipeline Runner - Windows Batch Script
REM Run: run_pipeline.bat

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║            MULTILINGUAL BOT PIPELINE - WINDOWS BATCH               ║
echo ║          Scrape URLs ^> Process Markdown ^> Upload to Supabase      ║
echo ╚════════════════════════════════════════════════════════════════════╝

echo.
echo 📋 Checking Prerequisites...
if not exist "links.txt" echo ⚠️  links.txt not found
if not exist "scraper.py" echo ⚠️  scraper.py not found
if not exist "uploader.py" echo ⚠️  uploader.py not found
if not exist "backend\.env" echo ⚠️  backend\.env not found

echo.
echo Step 1: Installing Dependencies...
python -m pip install -q -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install dependencies
    echo.
    echo Please try manually:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo ✅ Dependencies installed

echo.
echo Step 2: Scraping URLs to Markdown Files...
python scraper.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Scraping failed
    pause
    exit /b 1
)
echo ✅ Scraping complete

echo.
echo Step 3: Processing ^& Uploading to Supabase...
python uploader.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Upload failed
    pause
    exit /b 1
)
echo ✅ Upload complete

echo.
echo ╔════════════════════════════════════════════════════════════════════╗
echo ║                   ✅ PIPELINE COMPLETE!                            ║
echo ╠════════════════════════════════════════════════════════════════════╣
echo ║                                                                    ║
echo ║  ✓ Web content scraped from links.txt                             ║
echo ║  ✓ Markdown processed with semantic chunking                      ║
echo ║  ✓ Data uploaded to Supabase (embeddings table)                   ║
echo ║                                                                    ║
echo ║  🤖 Next: Start the chat API                                       ║
echo ║     npm run dev                                                     ║
echo ║     Then visit http://localhost:3000                               ║
echo ║                                                                    ║
echo ╚════════════════════════════════════════════════════════════════════╝

pause

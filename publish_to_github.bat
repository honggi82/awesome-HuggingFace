@echo off
setlocal
cd /d "%~dp0"

gh auth status
if errorlevel 1 (
  echo.
  echo GitHub login is required. Run:
  echo   gh auth login --hostname github.com --web --scopes repo
  exit /b 1
)

gh repo view honggi82/awesome-HuggingFace >nul 2>nul
if errorlevel 1 (
  gh repo create honggi82/awesome-HuggingFace --public --description "Awesome Hugging Face Papers: HF Daily Papers archive, 2023-05 to 2026-06" --source . --remote origin --push
) else (
  git remote remove origin >nul 2>nul
  git remote add origin https://github.com/honggi82/awesome-HuggingFace.git
  git push -u origin main
)
if errorlevel 1 exit /b %errorlevel%

gh api repos/honggi82/awesome-HuggingFace/pages -X POST -f "source[branch]=main" -f "source[path]=/docs" >nul 2>nul
if errorlevel 1 (
  gh api repos/honggi82/awesome-HuggingFace/pages -X PUT -f "source[branch]=main" -f "source[path]=/docs" >nul 2>nul
)

echo.
echo Done: https://github.com/honggi82/awesome-HuggingFace
echo Pages: https://honggi82.github.io/awesome-HuggingFace/

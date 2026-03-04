$git = "git"
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
   $git = "C:\Program Files\Git\cmd\git.exe"
}
& $git add .
& $git commit -m "Final documentation cleanup, LangChain integration restore, and Spanish translations"
& $git push

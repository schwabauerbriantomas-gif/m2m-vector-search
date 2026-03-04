$git = "git"
if (-not (Get-Command "git" -ErrorAction SilentlyContinue)) {
   $git = "C:\Program Files\Git\cmd\git.exe"
}
& $git add .
& $git commit -m "Fix remaining Apache 2.0 license references to AGPLv3"
& $git push

$ErrorActionPreference = "Stop"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\SchoolApplicationOCR"
$desktopDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
$startMenuDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
$shortcuts = @(
    (Join-Path $desktopDir "School Application OCR.lnk"),
    (Join-Path $startMenuDir "School Application OCR.lnk")
)

foreach ($shortcut in $shortcuts) {
    if (Test-Path -LiteralPath $shortcut -PathType Leaf) {
        Remove-Item -LiteralPath $shortcut -Force
    }
}
if (Test-Path -LiteralPath $installDir -PathType Container) {
    Remove-Item -LiteralPath $installDir -Recurse -Force
}
Write-Host "Application removed. Documents\SchoolApplicationOCR was not changed."

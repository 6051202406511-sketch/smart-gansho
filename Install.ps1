$ErrorActionPreference = "Stop"

try {
    $sourceExe = Join-Path $PSScriptRoot "SchoolApplicationOCR.exe"
    if (-not (Test-Path -LiteralPath $sourceExe -PathType Leaf)) {
        throw "SchoolApplicationOCR.exe was not found. Extract the full ZIP before installing."
    }

    $installDir = Join-Path $env:LOCALAPPDATA "Programs\SchoolApplicationOCR"
    $desktopDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    $startMenuDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
    $installedExe = Join-Path $installDir "SchoolApplicationOCR.exe"
    New-Item -ItemType Directory -Force -Path $installDir | Out-Null
    Copy-Item -LiteralPath $sourceExe -Destination $installedExe -Force

    $shell = New-Object -ComObject WScript.Shell
    foreach ($shortcutPath in @(
        (Join-Path $desktopDir "School Application OCR.lnk"),
        (Join-Path $startMenuDir "School Application OCR.lnk")
    )) {
        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $installedExe
        $shortcut.WorkingDirectory = $installDir
        $shortcut.Description = "Admissions document OCR"
        $shortcut.Save()
    }

    Write-Host "Installed to: $installDir"
    Write-Host "Your Excel files are kept in Documents\SchoolApplicationOCR."
}
catch {
    Write-Error $_
    exit 1
}

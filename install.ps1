$ErrorActionPreference = "Stop"

$Repo = "thewulf7/localcode"
$BinName = "localcode.exe"
$InstallDir = "$env:USERPROFILE\.localcode\bin"

# Create the install directory if it doesn't exist
if (-not (Test-Path -Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

Write-Host "Fetching latest release from GitHub for $Repo..."
$ApiUrl = "https://api.github.com/repos/$Repo/releases/latest"

try {
    $Release = Invoke-RestMethod -Uri $ApiUrl
}
catch {
    Write-Error "Failed to fetch latest release. Please verify the repository exists and is public."
    Exit 1
}

$DownloadUrl = $null
foreach ($Asset in $Release.assets) {
    # Check for windows specific binary
    if ($Asset.name -match "windows") {
        $DownloadUrl = $Asset.browser_download_url
        break
    }
}

if (-not $DownloadUrl) {
    # Fallback to any .exe file
    foreach ($Asset in $Release.assets) {
        if ($Asset.name -match "\.exe$") {
            $DownloadUrl = $Asset.browser_download_url
            break
        }
    }
}

if (-not $DownloadUrl) {
    Write-Error "Could not find a suitable Windows binary (.exe) in the latest release."
    Write-Host "You can still install using cargo: cargo install --git https://github.com/$Repo.git"
    Exit 1
}

$InstallPath = Join-Path -Path $InstallDir -ChildPath $BinName

Write-Host "Downloading $BinName from $DownloadUrl..."
Invoke-WebRequest -Uri $DownloadUrl -OutFile $InstallPath

Write-Host "✅ $BinName installed successfully to $InstallPath."

# Add to User PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notmatch [regex]::Escape($InstallDir)) {
    Write-Host "Adding $InstallDir to your PATH..."
    $NewPath = "$InstallDir;$UserPath"
    [Environment]::SetEnvironmentVariable("PATH", $NewPath, "User")
    
    # Update current session PATH so it works immediately
    $env:PATH = "$InstallDir;$env:PATH"
    
    Write-Host "⚠️ You may need to restart your terminal for the PATH changes to take effect in new windows." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Run 'localcode --help' to get started."

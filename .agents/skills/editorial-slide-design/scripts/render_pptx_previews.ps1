param(
    [Parameter(Mandatory=$true)][string]$PptxPath,
    [Parameter(Mandatory=$true)][string]$OutDir
)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open((Resolve-Path $PptxPath).Path, $true, $false, $false)
for ($i = 1; $i -le $pres.Slides.Count; $i++) {
    $pres.Slides.Item($i).Export((Join-Path $OutDir ("slide_{0}.png" -f $i)), "PNG", 1920, 1080)
}
$pres.Close()
$ppt.Quit()

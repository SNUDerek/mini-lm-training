$ErrorActionPreference = "Stop"

$outDir = "british_library_books_1800_1899"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Set-Location $outDir

$downloads = @(
    @{ Name = "1800_1809.zip"; Url = "https://bl.iro.bl.uk/downloads/91ae15cb-e08f-4abf-8396-e4742d9d4e37?locale=en" },
    @{ Name = "1810_1819.zip"; Url = "https://bl.iro.bl.uk/downloads/6d1a6e17-f28d-45b9-8f7a-a03cf3a96491?locale=en" },
    @{ Name = "1820_1829.zip"; Url = "https://bl.iro.bl.uk/downloads/ec764dbd-1ed4-4fc2-8668-b4df5c8ec451?locale=en" },
    @{ Name = "1830_1839.zip"; Url = "https://bl.iro.bl.uk/downloads/eab68022-0418-4df7-a401-78972514ed20?locale=en" },
    @{ Name = "1840_1849.zip"; Url = "https://bl.iro.bl.uk/downloads/d16d88b0-aa3f-4dfe-b728-c58d168d7b4d?locale=en" },
    @{ Name = "1850_1859.zip"; Url = "https://bl.iro.bl.uk/downloads/a6a44ea8-8d33-4880-8b17-f89c90e3d89a?locale=en" },
    @{ Name = "1860_1869.zip"; Url = "https://bl.iro.bl.uk/downloads/2e17f00f-52e6-4259-962c-b88ad60dec23?locale=en" },
    @{ Name = "1870_1879.zip"; Url = "https://bl.iro.bl.uk/downloads/899c3719-030c-4517-abd3-b28fdc85eed4?locale=en" },
    @{ Name = "1880_1889.zip"; Url = "https://bl.iro.bl.uk/downloads/ec3b8545-775b-47bd-885d-ce895263709e?locale=en" },
    @{ Name = "1890_1899.zip"; Url = "https://bl.iro.bl.uk/downloads/54ed2842-089a-439a-b751-2179b3ffba28?locale=en" }
)

foreach ($item in $downloads) {
    Write-Host "Downloading $($item.Name)..."

    curl.exe `
        --location `
        --continue-at - `
        --retry 5 `
        --retry-delay 10 `
        --fail `
        --output $item.Name `
        $item.Url
}
$files = Get-ChildItem `
    gene\datasets\knowledge_precursor `
    -Recurse `
    -File `
    -Filter "*.jsonl"

foreach ($file in $files) {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $file.FullName
    Write-Host "SIZE:" ([math]::Round($file.Length / 1MB, 2)) "MB"
    Write-Host "============================================================"

    $count = 0

    Get-Content `
        $file.FullName `
        -ReadCount 1 |
        ForEach-Object {

            if ($count -ge 2) {
                break
            }

            $line = $_

            try {

                $obj = $line | ConvertFrom-Json

                Write-Host ""
                Write-Host "RECORD $($count + 1) FIELDS:"
                $obj.PSObject.Properties.Name |
                    ForEach-Object {
                        Write-Host " -" $_
                    }

                Write-Host ""
                Write-Host "SAMPLE:"
                $obj | ConvertTo-Json -Depth 8

            }
            catch {

                Write-Host "NOT VALID JSON:"
                Write-Host $line
            }

            $count++
        }
}

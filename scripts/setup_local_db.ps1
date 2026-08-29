<#
.SYNOPSIS
    One-time local database setup.

.DESCRIPTION
    Creates the `orch` role and the `orch` / `orch_test` databases on a local
    PostgreSQL install, so the application never needs your superuser password.

    Prompts for the postgres superuser password you chose during installation.
    That password is used only for this session and is never written to disk.

    Run this in a normal PowerShell window (it prompts, so it cannot run inside a
    non-interactive tool session):

        powershell -ExecutionPolicy Bypass -File scripts\setup_local_db.ps1

.NOTES
    The `orch` password is a throwaway local development credential. It is the
    default in .env.example and is not a secret.
#>
[CmdletBinding()]
param(
    [string]$PgBin = "",
    [string]$PgHost = "localhost",
    [int]$Port = 5432,
    [string]$Superuser = "postgres",
    [string]$AppUser = "orch",
    [string]$AppPassword = "orch"
)

$ErrorActionPreference = "Stop"

if (-not $PgBin) {
    $candidate = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $candidate) {
        throw "Could not find psql.exe. Pass -PgBin 'C:\Path\To\PostgreSQL\16\bin'."
    }
    $PgBin = Split-Path $candidate.FullName -Parent
}
$psql = Join-Path $PgBin "psql.exe"
Write-Host "Using $psql" -ForegroundColor DarkGray

$secure = Read-Host -Prompt "Password for PostgreSQL superuser '$Superuser'" -AsSecureString
$env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
)

try {
    # CREATE ROLE / CREATE DATABASE cannot run inside a transaction block, and
    # neither supports IF NOT EXISTS, so each is guarded by a catalog lookup and
    # executed separately. Re-running the script is therefore safe.
    $roleExists = & $psql -h $PgHost -p $Port -U $Superuser -d postgres -tAc `
        "SELECT 1 FROM pg_roles WHERE rolname = '$AppUser'"
    if ($LASTEXITCODE -ne 0) { throw "Could not connect as '$Superuser'. Wrong password?" }

    if ($roleExists -ne "1") {
        & $psql -h $PgHost -p $Port -U $Superuser -d postgres -c `
            "CREATE ROLE $AppUser LOGIN PASSWORD '$AppPassword' CREATEDB"
        if ($LASTEXITCODE -ne 0) { throw "Failed to create role $AppUser" }
        Write-Host "created role  $AppUser" -ForegroundColor Green
    } else {
        # Keep the password in sync with .env.example on a re-run.
        & $psql -h $PgHost -p $Port -U $Superuser -d postgres -c `
            "ALTER ROLE $AppUser LOGIN PASSWORD '$AppPassword' CREATEDB" | Out-Null
        Write-Host "role          $AppUser (already existed, password reset)" -ForegroundColor DarkGray
    }

    foreach ($db in @("orch", "orch_test")) {
        $dbExists = & $psql -h $PgHost -p $Port -U $Superuser -d postgres -tAc `
            "SELECT 1 FROM pg_database WHERE datname = '$db'"
        if ($dbExists -ne "1") {
            & $psql -h $PgHost -p $Port -U $Superuser -d postgres -c `
                "CREATE DATABASE $db OWNER $AppUser"
            if ($LASTEXITCODE -ne 0) { throw "Failed to create database $db" }
            Write-Host "created db    $db" -ForegroundColor Green
        } else {
            Write-Host "database      $db (already existed)" -ForegroundColor DarkGray
        }
    }
} finally {
    Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Done. Connection string:" -ForegroundColor Cyan
Write-Host "  postgresql+psycopg://${AppUser}:${AppPassword}@${PgHost}:${Port}/orch"
Write-Host ""
Write-Host "Verify with:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\python.exe -m alembic upgrade head"

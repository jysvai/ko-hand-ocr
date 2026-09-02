# 학습을 배경으로 띄우되, **첫 걸음이 실제로 지나갈 때까지 확인**하고 넘어간다.
#
#   powershell -File tools/train.ps1 -Resume runs/v4 -Out runs/v5
#
# 왜 필요한가. 앞의 CUDA 프로세스를 강제 종료한 뒤 곧바로 새로 띄우면
# 첫 backward 에서 드라이버가 메모리를 안 내준다.
#
#   torch.AcceleratorError: CUDA error: out of memory        <- VRAM 은 8GB 다 비어 있다
#   RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR   <- 같은 뿌리
#
# nvidia-smi 는 이미 0 MiB 를 찍고 있어서 비었다고 착각하기 쉽다. 실제로는
# 정리가 덜 끝난 것이다. 게다가 이때 죽는 모양이 파이썬 Traceback 이 아니라
# **네이티브 스택 덤프**라, 로그를 Traceback 으로만 거르면 조용히 놓친다.
#
# 그래서 여기서는 (1) 남은 python 이 없을 때까지 기다리고 (2) 띄운 뒤
# 첫 손실 줄을 볼 때까지 지켜보고 (3) 실패하면 쉬었다 다시 띄운다.
#
# -Lr / -Keep 을 밖으로 뺀 까닭. 합성 분포를 크게 바꾼 뒤에는 **짧게 돌고
# 중간 판을 촘촘히 남겨야** 한다. 실측: 분포를 바꾼 뒤 15,000걸음이 꼭짓점이고
# 30,000걸음은 오히려 내려갔다(88.1 -> 84.3), 45,000걸음도 못 따라잡았다(86.0).
# `--resume` 은 학습률 일정을 처음부터 되돌리므로, 이어 달릴 때는 꼭대기
# 학습률을 낮춰서 이미 잘하는 것을 흔들지 않는다.

param(
  [string]$Resume = "",
  [string]$Out = "runs/next",
  [int]$Steps = 45000,
  [int]$Keep = 15000,
  [double]$Lr = 3e-4,
  [double]$EncoderLr = 3e-5,
  [int]$Warmup = 1000,
  [int]$Batch = 40,
  [int]$Workers = 3,
  # 일꾼 수를 5 -> 3 으로 내렸다. 5 로 두면 첫 backward 에서 'CUDA error: out of
  # memory' 가 나는데 **GPU 문제가 아니다** — nvidia-smi 로 7,043 MiB 가 비어
  # 있고 작은 backward 는 멀쩡히 된다. 고정(pinned) 메모리는 호스트 RAM 이고,
  # 합성이 무거워지면서(획 굵기 상향, _loosen) 일꾼 다섯이 호스트를 밀어냈다.
  # 3 이면 확실히 돈다. 되띄우기 5번으로도 못 넘긴 것이 이것이었다.
  [int]$Letters = 128,
  # 경로를 코드에 박지 않는다. 만든 사람의 PC 경로가 박혀 있으면 공개했을 때
  # 남의 PC 에서 안 돌고 계정 이름도 같이 나간다.
  #   글꼴  : $env:KOHAND_FONTS  없으면 ~/.cache/ko-hand-ocr/fonts
  #   파이썬: $env:KOHAND_PYTHON 없으면 지금 PATH 의 python
  [string]$Fonts = $(if ($env:KOHAND_FONTS) { $env:KOHAND_FONTS }
                     else { Join-Path $env:USERPROFILE ".cache/ko-hand-ocr/fonts" }),
  [string]$Python = $(if ($env:KOHAND_PYTHON) { $env:KOHAND_PYTHON }
                      else { (Get-Command python).Source }),
  [int]$Tries = 5
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
# expandable_segments 는 Windows 에서 안 먹는다(torch 가 경고만 내고 무시한다).
# 첫 backward 의 out of memory 는 조각 문제가 아니라 **호스트 RAM** 이었다.
# 고정(pinned) 메모리는 호스트에 잡히고, 합성이 무거워지자 일꾼 다섯이 넘쳤다.
# 고침은 -Workers 3 이다(아래 param 을 보라).
$log = "$Out.log"
$err = "$Out.err"

function Wait-ForQuietGpu {
  # 남은 python 을 정리하고, GPU 가 실제로 놓아 줄 때까지 기다린다.
  #
  # 기다리는 조건을 **'python 이 하나도 없을 때'로 두면 안 된다.** 부모가 죽은
  # 뒤 남은 일꾼(spawn_main)은 안 죽고 0MB 로 남는 일이 있는데, 그러면 이 조건은
  # 영영 참이 되지 않는다. 실제로 그래서 두 번 다 '그래도 해 본다'로 빠져나와
  # **학습이 둘 동시에 떴고**, 먼저 뜬 쪽이 GPU 를 쥔 채 2GB 일꾼 셋을 돌려서
  # 나중 것이 첫 backward 를 못 넘겼다. 로그에는 그냥 '첫 걸음을 못 지났다'로만
  # 보여서, 원인을 GPU 드라이버 쪽으로 잘못 찾게 된다.
  #
  # 그러니 (1) 죽일 것은 **학습 프로세스**이고 (2) 기다릴 것은 **GPU 메모리**다.
  # 못 죽는 껍데기는 GPU 를 안 쥐고 있으니 세지 않는다.
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "kohandocr" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $PID -and $_.WorkingSet64 -gt 200MB } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
  for ($i = 0; $i -lt 60; $i++) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -and $_.CommandLine -match "kohandocr" }).Count
    $used = [int](@(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)[0])
    if ($busy -eq 0 -and $used -lt 400) { Start-Sleep -Seconds 3; return }
    Start-Sleep -Seconds 2
  }
  Write-Output "  (GPU 가 계속 바쁘다. 그래도 해 본다)"
}

$args = @("-u", "-m", "kohandocr.train",
          "--fonts", $Fonts, "--out", $Out,
          "--steps", $Steps, "--batch", $Batch, "--workers", $Workers,
          "--letters", $Letters,
          "--lr", $Lr, "--encoder-lr", $EncoderLr, "--warmup", $Warmup,
          "--log-every", "500", "--check-every", "2500", "--save-every", "2500",
          "--keep", $Keep)
if ($Resume) { $args += @("--resume", $Resume) }

for ($try = 1; $try -le $Tries; $try++) {
  Wait-ForQuietGpu
  Remove-Item $log, $err -Force -ErrorAction SilentlyContinue
  $proc = Start-Process -FilePath $Python -ArgumentList $args `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent) `
    -RedirectStandardOutput $log -RedirectStandardError $err `
    -NoNewWindow -PassThru
  Write-Output "$try 번째 시도: pid $($proc.Id)"

  # 첫 손실 줄이 나올 때까지 지켜본다. 한글로 찾으면 깨지므로 ASCII 조각("lr 0.00")으로 찾는다.
  $bad = "out of memory|CUBLAS|Unhandled exception|AbortHandler|Traceback"
  for ($i = 0; $i -lt 120; $i++) {
    if ($proc.HasExited) { break }
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "lr [0-9]" -Quiet)) {
      # 첫 손실 줄을 봤다고 산 것이 아니다. **v21 이 그렇게 죽었다** — 1걸음을
      # 찍고 곧바로 첫 backward 에서 out of memory 로 넘어갔는데, 여기서 exit 0
      # 을 해 버려서 '돈다'로 보고했다. 중간 판이 하나도 안 남아 10,000걸음을
      # 통째로 잃었고, 죽은 줄도 한참 뒤에 알았다.
      #
      # 그래서 첫 줄을 본 뒤 **30초 더 지켜본다.** 호스트 RAM 이 모자라 죽는
      # 것은 늘 이 안에서 터진다(일꾼들이 첫 묶음을 채우는 때다).
      $first = (Select-String -Path $log -Pattern "lr [0-9]" | Select-Object -First 1).Line
      for ($k = 0; $k -lt 15; $k++) {
        Start-Sleep -Seconds 2
        if ($proc.HasExited) { break }
        if ((Test-Path $err) -and (Select-String -Path $err -Pattern $bad -Quiet)) { break }
      }
      if (-not $proc.HasExited -and
          -not ((Test-Path $err) -and (Select-String -Path $err -Pattern $bad -Quiet))) {
        Write-Output "돈다: $($first.Trim())"
        exit 0
      }
      Write-Output "  첫 걸음은 지났는데 곧 죽었다. 일꾼 수를 줄여야 한다."
    }
    if ((Test-Path $err) -and (Select-String -Path $err -Pattern $bad -Quiet)) { break }
    Start-Sleep -Seconds 2
  }

  Write-Output "  첫 걸음을 못 지났다. 쉬었다 다시 띄운다."
  if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 20
}

Write-Output "$Tries 번 다 실패했다. tools/train.ps1 의 머리말을 보라."
exit 1

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
  # 있고 작은 backward 는 멀쩡히 된다. 모자란 것은 호스트 쪽이다.
  #
  # **다시 5 로 올려 보고 또 되돌렸다.** 이번엔 이유를 정확히 잡았다:
  # 실제 메모리가 아니라 **커밋(commit charge)** 이다. 일꾼 하나가 실제로
  # 만지는 것은 1GB 인데 **커밋은 7.9GB** 를 잡는다. 이 PC 는 RAM 23.3GB,
  # 커밋 한도 62GB 라 일꾼 5(39.5GB) + 본체(7GB) 면 한도의 99% 에 붙는다.
  # 그러면 학습이 1걸음에서 몇 분씩 멈춘다. 게다가 속도 이득도 없었다 —
  # 3일꾼 72장/초, 5일꾼 73장/초다(단독으로 재도 그렇다).
  # **이 PC 에서 3 이 상한이다.** 더 빠르게 하려면 RAM 을 꽂아야 한다.
  [int]$Seed = 0,
  # 씨앗. **여태 안 넘기고 있었다** — train.py 에 --seed 가 있는데 여기서
  # 안 주니 모든 판이 씨앗 0 이었다. 앙상블은 식구가 서로 다를수록 이득인데,
  # 그 가장 싼 축(첫 무게와 자료 순서)을 안 쓰고 있었다.
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
  # 죽일 것은 **딱 둘**이다: 학습 본체(kohandocr.train)와, 부모가 사라진 일꾼
  # (spawn_main). 그 밖에는 건드리지 않는다.
  #
  # **크기로 고르면 안 된다.** 한 번 그렇게 짰다가 크게 데었다 —
  # `PageFileUsage -gt 500MB` 로 걸렀더니, 이 스크립트를 부른 `tools/loop.py`
  # 가 torch 를 올려 500MB 를 넘는 바람에 **고리가 제 손에 죽었다.** v24 는
  # 20,000걸음을 다 돌았는데 그것을 채점할 놈이 없어서 하루가 헛돌았다.
  # `-ne $PID` 로는 못 막는다. 여기서 $PID 는 PowerShell 자신이지 고리가 아니다.
  #
  # `kohandocr` 로만 거르는 것도 모자란다. 일꾼(spawn_main)의 명령줄에는 그
  # 글자가 없어서, 부모가 죽으면 아무 조건에도 안 걸리고 커밋만 문 채 남는다.
  $all = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")
  $ids = @($all | Select-Object -ExpandProperty ProcessId)
  $all | Where-Object {
      ($_.CommandLine -and $_.CommandLine -match "kohandocr\.train") -or
      ($_.CommandLine -and $_.CommandLine -match "spawn_main" -and
       $ids -notcontains $_.ParentProcessId)
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
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
          "--letters", $Letters, "--seed", $Seed,
          "--lr", $Lr, "--encoder-lr", $EncoderLr, "--warmup", $Warmup,
          "--log-every", "500", "--check-every", "2500", "--save-every", "2500",
          "--keep", $Keep)
if ($Resume) { $args += @("--resume", $Resume) }
# 값에 공백이 있으면 감싼다. 위 프로브와 같은 함정이다 — 글꼴 폴더 이름에
# 빈칸이 하나만 있어도 Start-Process 가 두 인자로 쪼개서 엉뚱한 곳을 뒤진다.
$args = $args | ForEach-Object { if ("$_" -match '\s') { '"' + $_ + '"' } else { $_ } }

# **되띄우기로 못 고치는 것은 먼저 걸러 낸다.** torch 가 없는 파이썬을 잡으면
# 다섯 번을 다시 띄워도 똑같이 죽는데, 로그에는 '첫 걸음을 못 지났다'로만 보여서
# GPU 문제로 잘못 찾게 된다. 실제로 그랬다 — $Python 기본값을 계정 경로에서
# 떼면서 시스템 파이썬(torch 없음)을 잡았다.
# Start-Process 로 부른다. 그냥 `& $Python ...` 을 쓰면 $ErrorActionPreference
# = "Stop" 이 네이티브 명령의 stderr 를 오류로 바꿔서 **검사 자체가 터진다.**
# 인용부호를 **직접** 넣는다. `-ArgumentList @("-c", "import torch, transformers")`
# 로 적으면 PowerShell 이 배열을 따옴표 없이 공백으로 이어 붙여서, 파이썬은
# `-c import` 만 받고 SyntaxError 로 죽는다. 그러면 이 검사가 **멀쩡한 파이썬을
# 'torch 가 없다'고 몰아세운다** — 실제로 그래서 v24 를 못 띄웠다.
$probe = Start-Process -FilePath $Python -ArgumentList @("-c", '"import torch, transformers"') `
  -NoNewWindow -Wait -PassThru -RedirectStandardError ([System.IO.Path]::GetTempFileName())
if ($probe.ExitCode -ne 0) {
  Write-Output "이 파이썬에는 torch/transformers 가 없다: $Python"
  Write-Output "  `$env:KOHAND_PYTHON 을 정하거나 -Python 으로 넘겨라."
  exit 1
}

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

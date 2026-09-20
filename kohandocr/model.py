"""ko-hand-ocr 모델.

    그림 -> ViT 인코더 -> 자모 디코더 -> 글월

**인코더**는 ImageNet 으로 미리 배운 ViT-small 을 가져다 쓴다.
(`facebook/deit-small-patch16-224`, Apache-2.0. 이름은 deit 인데 알맹이는 vit 체크포인트라
 `ViTModel` 로 열어야 가중치가 실린다. `DeiTModel` 로 열면 조용히 난수로 시작한다 — 실제로 겪었다.)
글씨가 아니라 사진의 결·모서리·획을 보는 눈이라 우리 일에도 그대로 쓸모가 있다.

**디코더**는 맨 처음부터 우리가 학습한다. 물려받은 가중치가 하나도 없으므로
학습 데이터의 이용약관을 물려받을 일도 없다. 이게 ko-trocr 를 안 쓰는 이유다.
자세한 것은 PROVENANCE.md 에 있다.

**입력은 정사각형이 아니다.** 글줄은 가로로 길다. 224x224 에 욱여넣으면
긴 문장이 뭉개진다. 그래서 64x640 (10:1) 을 쓰고, 사전학습 위치 임베딩을
14x14 격자에서 4x40 격자로 늘려 박는다. 매번 `interpolate_pos_encoding=True`
를 넘기는 방법도 되지만, 한 번이라도 빠뜨리면 터지거나 조용히 틀린다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import TrOCRConfig, TrOCRForCausalLM, VisionEncoderDecoderModel, ViTModel

# 규격은 `cell.py` 에서 바로 가져온다. 예전에는 `synth` 를 거쳤는데, 그러면
# **읽기만 하려는 쪽도 합성 모듈 전체를 끌고 온다.** 오픈소스로 내보낼 때
# 읽기 경로는 글꼴도 합성 코드도 없이 돌아야 한다.
from .cell import CELL

ENCODER = "facebook/deit-small-patch16-224"      # Apache-2.0, ImageNet-1k


def _restretch(encoder: ViTModel, cell: tuple[int, int]) -> None:
    """위치 임베딩을 정사각 격자에서 글줄 모양 격자로 늘린다.

    격자 한 칸은 '그림의 이 자리' 라는 뜻이다. 14x14 로 배운 자리 감각을
    4x40 으로 늘리면 위아래는 성기어지고 좌우는 촘촘해진다. 글줄에 맞는 모양이다.
    """
    patch = encoder.config.patch_size
    rows, cols = cell[0] // patch, cell[1] // patch
    pos = encoder.embeddings.position_embeddings
    cls, grid = pos[:, :1], pos[:, 1:]
    side = int(round(grid.shape[1] ** 0.5))
    grid = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(rows, cols), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, rows * cols, -1)
    encoder.embeddings.position_embeddings = torch.nn.Parameter(torch.cat([cls, grid], dim=1))

    # 세 군데가 따로 크기를 들고 있다. 하나라도 빼먹으면 forward 에서 막힌다.
    encoder.config.image_size = list(cell)
    encoder.embeddings.image_size = cell
    encoder.embeddings.patch_embeddings.image_size = cell


def build(vocab, cell: tuple[int, int] = CELL, layers: int = 4, heads: int = 6,
          ffn: int = 1536, dropout: float = 0.1, letters: int = 64) -> VisionEncoderDecoderModel:
    """새 모델 한 벌. 인코더만 사전학습이고 디코더는 백지다."""
    encoder = ViTModel.from_pretrained(ENCODER, add_pooling_layer=False)
    _restretch(encoder, cell)

    decoder = TrOCRForCausalLM(TrOCRConfig(
        vocab_size=len(vocab),
        d_model=encoder.config.hidden_size,
        decoder_layers=layers,
        decoder_attention_heads=heads,
        decoder_ffn_dim=ffn,
        max_position_embeddings=letters,
        dropout=dropout,
        activation_dropout=dropout,
        attention_dropout=dropout,
        pad_token_id=vocab.pad,
        bos_token_id=vocab.bos,
        eos_token_id=vocab.eos,
        decoder_start_token_id=vocab.bos,
        scale_embedding=True,
        use_cache=True,
        use_learned_position_embeddings=True,
        layernorm_embedding=True,
    ))

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.config.decoder_start_token_id = vocab.bos
    model.config.pad_token_id = vocab.pad
    model.config.eos_token_id = vocab.eos
    model.generation_config.decoder_start_token_id = vocab.bos
    model.generation_config.pad_token_id = vocab.pad
    model.generation_config.eos_token_id = vocab.eos
    # `max_length` 가 아니라 `max_new_tokens` 로 둔다. 읽을 때는 `max_new_tokens`
    # 를 넘기는데, 둘이 같이 있으면 transformers 가 부를 때마다 경고를 낸다:
    # "Both `max_new_tokens` and `max_length` seem to have been set." 우리에게는
    # 새로 뽑는 자모 수가 맞는 뜻이다 — 앞머리 길이는 늘 1(시작 토큰)이라 값은
    # 같지만, 쓰는 사람이 부를 때마다 볼 소리를 남길 까닭이 없다.
    model.generation_config.max_new_tokens = letters
    return model


def load(folder) -> VisionEncoderDecoderModel:
    """저장해 둔 모델을 연다. 크기 정보는 저장된 config 에 이미 들어 있다.

    불러올 때 `encoder.pooler.dense` 가 MISSING 이라고 찍히는데 괜찮다.
    우리는 pooler 없이 만들었고(`add_pooling_layer=False`),
    VisionEncoderDecoder 는 마지막 층의 값만 쓰지 pooler 를 보지 않는다.
    """
    model = VisionEncoderDecoderModel.from_pretrained(str(folder))
    # 예전에 저장한 판은 `max_length` 로 적혀 있다. 그대로 두면 읽을 때마다
    # 경고가 나므로 여기서 옮긴다. 받은 사람이 손볼 일이 아니다.
    spec = model.generation_config
    if getattr(spec, "max_length", None) and not getattr(spec, "max_new_tokens", None):
        spec.max_new_tokens, spec.max_length = spec.max_length, None
    encoder = model.encoder
    encoder.embeddings.image_size = tuple(encoder.config.image_size)
    encoder.embeddings.patch_embeddings.image_size = tuple(encoder.config.image_size)
    return model


def grow(model: VisionEncoderDecoderModel, layers: int) -> int:
    """디코더를 `layers` 층까지 **이어받은 채로** 늘린다. 늘린 층 수를 돌려준다.

    맨바닥에서 큰 모델을 새로 배우지 않는 까닭. 씨앗만 바꾼 맨바닥 판 넷
    (v27~v30)이 모두 글씨체 평균 87%대에 멈춰 이어받은 판(93%)을 한 번도 못
    이겼다. 자리를 늘리더라도 **지금 아는 것을 들고** 늘려야 한다.

    인코더와 입력 크기는 건드리지 않는다. 읽는 시간의 대부분은 자모를 하나씩
    뽑는 오버헤드라 디코더 층을 늘려도 느려지는 몫이 작고, 인코더를 키우면
    31M 이라는 이 모델의 성격(CPU 만으로 칸당 0.16초)이 먼저 무너진다.

    새 층은 맨 끝 층을 복제하되 **더하는 몫(attention 출력, fc2)을 0** 으로
    두고, 가운데 LayerNorm 둘은 항등(1, 0)으로, 층 끝 LayerNorm 은 맨 끝 층 것을
    물려받는다. 답을 그대로 지키지는 못한다 — TrOCR 층은 더한 뒤에 정규화하는
    post-LN 이라 더하는 몫이 0 이어도 LayerNorm 이 한 번 더 걸린다. 그래서 재서
    골랐다(2026-09-17, v46 을 6층으로, 합성 48줄, CPU):

        그대로 4층        손실 0.052
        0 으로 늘림       손실 0.060   그리디 답 8줄 중 7줄 그대로
        끝 층 복제        손실 2.585   그리디 답 8줄 중 0줄 — 통째로 무너진다
    """
    import copy

    decoder = model.decoder.model.decoder
    have = len(decoder.layers)
    if layers <= have:
        return 0
    last = decoder.layers[-1]
    for at in range(have, layers):
        fresh = copy.deepcopy(last)
        for lin in (fresh.self_attn.out_proj, fresh.encoder_attn.out_proj, fresh.fc2):
            torch.nn.init.zeros_(lin.weight)
            if lin.bias is not None:
                torch.nn.init.zeros_(lin.bias)
        for norm in (fresh.self_attn_layer_norm, fresh.encoder_attn_layer_norm):
            torch.nn.init.ones_(norm.weight)
            torch.nn.init.zeros_(norm.bias)
        # 캐시 자리 번호. 복제한 층은 맨 끝 층의 번호를 들고 있어서, 그대로 두면
        # 읽을 때 두 층이 **같은 캐시 칸**에 키·값을 덮어쓴다.
        for attn in (fresh.self_attn, fresh.encoder_attn):
            if hasattr(attn, "layer_idx"):
                attn.layer_idx = at
        if hasattr(fresh, "layer_idx"):
            fresh.layer_idx = at
        decoder.layers.append(fresh)
    model.decoder.config.decoder_layers = layers
    model.config.decoder.decoder_layers = layers
    return layers - have


def sizes(model: VisionEncoderDecoderModel) -> dict[str, float]:
    million = 1e6
    return {
        "전체": sum(p.numel() for p in model.parameters()) / million,
        "인코더": sum(p.numel() for p in model.encoder.parameters()) / million,
        "디코더": sum(p.numel() for p in model.decoder.parameters()) / million,
    }

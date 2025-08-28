# losses/truncated_smoothap_cfg.py

truncated_smoothap_cfg = dict(
    type='TruncatedSmoothAP',  # 이를 통해 create_task나 손실 함수 팩토리에서 로드
    tau1=0.01,                 # 스무딩 상수 (예: MinkLoc3Dv2 원본과 동일)
    k_top=None,                # top-k truncation (원본에서 필요시 사용)
    margin=None                # margin (해당 Loss에서는 사용 안함; None)
)

import unittest

try:
    import torch
    from easydict import EasyDict

    from method_core.bimamba import BiMambaEncoderLayer, Mamba
    from method_core.model import VideoGARModel
except ImportError:
    torch = None
    Mamba = None


GPU_READY = torch is not None and Mamba is not None and torch.cuda.is_available()


@unittest.skipUnless(GPU_READY, "requires CUDA, PyTorch, easydict, and mamba-ssm")
class VideoGARGPUSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        cls.device = torch.device("cuda:0")
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

    def test_bimamba_forward(self):
        layer = (
            BiMambaEncoderLayer(
                hidden_size=32,
                dropout=0.0,
                d_state=4,
                d_conv=3,
                expand=1,
                fuse_mode="sum",
            )
            .to(self.device)
            .eval()
        )
        inputs = torch.randn(2, 12, 32, device=self.device)
        mask = torch.tensor(
            [[1] * 12, [1] * 9 + [0] * 3], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            outputs = layer(inputs, mask)

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(torch.isfinite(outputs).all())
        self.assertEqual(torch.count_nonzero(outputs[1, 9:]).item(), 0)

    def test_training_and_inference_forward(self):
        config = EasyDict(
            ctx_mode="video",
            max_desc_l=8,
            max_ctx_l=12,
            hidden_size=32,
            input_drop=0.0,
            drop=0.0,
            query_input_size=16,
            visual_input_size=20,
            sub_input_size=16,
            n_heads=4,
            conv_kernel_size=3,
            conv_stride=1,
            initializer_range=0.02,
            backbone_type="BiMamba",
            mamba_d_state=4,
            mamba_d_conv=3,
            mamba_expand=1,
            mamba_fuse_mode="sum",
            use_generative_augmentation=True,
            use_fusion_encoder=True,
            fusion_num_layers=1,
            lm_weight=0.3,
            lm_pad_token_id=0,
            lm_vocab_size=101,
            lm_num_layers=1,
            lw_fcl=0.0,
            lw_vcl=0.0,
            lw_st_ed=1.0,
            lw_neg_ctx=0.0,
            lw_neg_q=0.0,
            margin=0.1,
            ranking_loss_type="hinge",
            hard_pool_size=2,
            use_hard_negative=False,
        )
        model = VideoGARModel(config).to(self.device).eval()
        query_feat = torch.randn(2, 8, 16, device=self.device)
        query_mask = torch.tensor(
            [[1] * 8, [1] * 6 + [0] * 2], dtype=torch.float32, device=self.device
        )
        video_feat = torch.randn(2, 12, 20, device=self.device)
        video_mask = torch.tensor(
            [[1] * 12, [1] * 10 + [0] * 2], dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            loss, _ = model(
                query_feat=query_feat,
                query_mask=query_mask,
                video_feat=video_feat,
                video_mask=video_mask,
                sub_feat=None,
                sub_mask=None,
                st_ed_indices=torch.tensor([[2, 7], [1, 6]], device=self.device),
                match_labels=torch.zeros(2, 12, dtype=torch.long, device=self.device),
                query_input_ids=torch.randint(1, 101, (2, 8), device=self.device),
                query_attn_mask=query_mask.long(),
            )
            encoded_video, _ = model.encode_context(video_feat, video_mask, None, None)
            q2ctx, st_logits, ed_logits = model.get_pred_from_raw_query(
                query_feat,
                query_mask,
                encoded_video,
                video_mask,
                None,
                None,
                cross=False,
            )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(q2ctx.shape, (2, 2))
        self.assertEqual(st_logits.shape, (2, 12))
        self.assertEqual(ed_logits.shape, (2, 12))
        self.assertTrue(torch.isfinite(q2ctx).all())
        self.assertTrue(torch.isfinite(st_logits).all())
        self.assertTrue(torch.isfinite(ed_logits).all())


if __name__ == "__main__":
    unittest.main()

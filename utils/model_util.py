import torch
import os

from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    """Load a model state dict while ignoring missing positional encodings."""
    # keep compatibility with older checkpoints
    state_dict.pop('sequence_pos_encoder.pe', None)
    state_dict.pop('embed_timestep.sequence_pos_encoder.pe', None)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")


def create_model_and_diffusion(args, data, ModelClass=MDM, DiffusionClass=SpacedDiffusion):
    model = ModelClass(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion


def get_model_args(args, data):
    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)

    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml', '100style']:
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {
        'modeltype': '',
        'njoints': njoints,
        'nfeats': nfeats,
        'num_actions': num_actions,
        'translation': True,
        'pose_rep': data_rep,                 # <<< important: match data_rep
        'glob': True,
        'glob_rot': True,
        'latent_dim': args.latent_dim,
        'ff_size': 1024,
        'num_layers': args.layers,
        'num_heads': 4,
        'dropout': 0.1,
        'activation': "gelu",
        'data_rep': data_rep,
        'cond_mode': cond_mode,
        'cond_mask_prob': args.cond_mask_prob,
        'action_emb': action_emb,
        'arch': args.arch,
        'emb_trans_dec': args.emb_trans_dec,
        'clip_version': clip_version,
        'dataset': args.dataset,
        'emb_before_mask': args.emb_before_mask,
        'text_encoder_type': args.text_encoder_type,
        'pos_embed_max_len': args.pos_embed_max_len,
        'mask_frames': args.mask_frames,
        'lora_finetune': args.lora_finetune,
        'lora_rank': args.lora_rank,
        'lora_layer': args.lora_layer,
        'no_lora_q': args.no_lora_q,
        'lora_ff': args.lora_ff,
        'use_age': getattr(args, 'age_cond', False),
    }


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    print(f"number of diffusion-steps: {steps}")

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if hasattr(args, 'multi_train_mode'):
        multi_train_mode = args.multi_train_mode
    else:
        multi_train_mode = None

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
        model_var_type=((gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
                        if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_prior_preserv=args.lambda_prior_preserv
    )


def load_saved_model(model, model_path, use_avg: bool = False):
    state_dict = torch.load(model_path, map_location='cpu')
    if use_avg and 'model_avg' in state_dict.keys():
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model


def load_lora_to_model(model, lora_path, use_avg: bool = False):
    if '.pt' not in lora_path:
        lora_path = find_lora_path(lora_path)

    state_dict = torch.load(lora_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all(['lora' not in k or 'q_zero' in k for k in missing_keys])


def find_lora_path(style, base_path='save/lora'):
    for dir in os.listdir(base_path):
        model_path = os.path.join(base_path, dir, 'model000004000.pt')
        if style in dir and os.path.exists(model_path):
            return model_path
    raise Exception(f'lora for style {style} not found at {base_path}')
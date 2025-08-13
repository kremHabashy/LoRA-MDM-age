
from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            if a == 'diffusion_steps' and model_args[a] != args.diffusion_steps:
                print(f'diffusion_steps overwrite, diffusion_steps={model_args[a]}!!!')
            setattr(args, a, model_args[a])
        elif 'cond_mode' in model_args:
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')


def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_lora_options(parser):
    group = parser.add_argument_group('lora')
    group.add_argument("--lora_finetune", action='store_true')
    group.add_argument("--styles", default=None,  nargs='+', help='None for all styles, or list of styles to use.')
    group.add_argument("--lora_rank", default=5, type=int)
    group.add_argument("--lora_layer", default=-100, type=int, help='Transformers layer to use for lora, negative for all layers.')
    group.add_argument("--no_lora_q", action='store_true', help='remove lora adapter from query')
    group.add_argument("--lora_ff", action='store_true', help='add lora adapter to feed forward layers')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=200, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str)
    group.add_argument("--diffusion_steps", default=1000, type=int)
    group.add_argument("--sigma_small", default=True, type=bool)


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc', choices=['trans_enc', 'trans_dec', 'gru'], type=str)
    group.add_argument("--text_encoder_type", default='clip', choices=['clip', 'bert'], type=str)
    group.add_argument("--emb_trans_dec", default=False, type=bool)
    group.add_argument("--layers", default=8, type=int)
    group.add_argument("--latent_dim", default=512, type=int)
    group.add_argument("--cond_mask_prob", default=.1, type=float)
    group.add_argument("--mask_frames", action='store_true')
    group.add_argument("--lambda_rcxyz", default=0.0, type=float)
    group.add_argument("--lambda_vel", default=0.0, type=float)
    group.add_argument("--lambda_fc", default=0.0, type=float)
    group.add_argument("--lambda_prior_preserv", default=1.0, type=float)
    group.add_argument("--unconstrained", action='store_true')
    group.add_argument("--emb_before_mask", action='store_true')
    group.add_argument("--pos_embed_max_len", default=5000, type=int)
    group.add_argument("--use_ema", action='store_true')
    group.add_argument("--age_cond", action='store_true',
                       help="Enable age MLP conditioning.")
    # NEW: let user choose conditioning explicitly
    group.add_argument("--cond_mode", default=None, type=str,
                       help="Conditioning string, e.g. 'text', 'age', 'text+age', 'action', 'no_cond'.")


def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml',
                       choices=['humanml', 'kit', 'humanact12', 'uestc', '100style', 'vc'], type=str)
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str)
    group.add_argument("--overwrite", action='store_true')
    group.add_argument("--train_platform_type", default='NoPlatform',
                       choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str)
    group.add_argument("--lr", default=1e-5, type=float)
    group.add_argument("--weight_decay", default=0.0, type=float)
    group.add_argument("--lr_anneal_steps", default=0, type=int)
    group.add_argument("--eval_batch_size", default=32, type=int)
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str)
    group.add_argument("--eval_during_training", action='store_true')
    group.add_argument("--eval_rep_times", default=3, type=int)
    group.add_argument("--eval_num_samples", default=1_000, type=int)
    group.add_argument("--log_interval", default=1_000, type=int)
    group.add_argument("--save_interval", default=50_000, type=int)
    group.add_argument("--num_steps", default=600_000, type=int)
    group.add_argument("--num_frames", default=60, type=int)
    group.add_argument("--resume_checkpoint", default="", type=str)
    group.add_argument("--gen_during_training", action='store_true')
    group.add_argument("--gen_num_samples", default=3, type=int)
    group.add_argument("--gen_num_repetitions", default=2, type=int)
    group.add_argument("--gen_guidance_param", default=2.5, type=float)
    group.add_argument("--avg_model_beta", default=0.9999, type=float)
    group.add_argument("--adam_beta2", default=0.999, type=float)
    group.add_argument("--starting_checkpoint", default="", type=str)
    group.add_argument("--inpainting_mask", default=None, type=str)


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--lora_path", type=str)
    group.add_argument("--output_dir", default='', type=str)
    group.add_argument("--num_samples", default=10, type=int)
    group.add_argument("--num_repetitions", default=3, type=int)
    group.add_argument("--guidance_param", default=2.5, type=float)
    group.add_argument("--prompt_suffix", default=None)
    group.add_argument("--inpainting_mask", default=None, type=str)


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float)
    group.add_argument("--input_text", default='', type=str)
    group.add_argument("--action_file", default='', type=str)
    group.add_argument("--text_prompt", default='', type=str)
    group.add_argument("--action_name", default='', type=str)


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str)
    group.add_argument("--text_condition", default='', type=str)
    group.add_argument("--prefix_end", default=0.25, type=float)
    group.add_argument("--suffix_start", default=0.75, type=float)


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--lora_path", type=str)
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str)
    group.add_argument("--guidance_param", default=2.5, type=float)
    group.add_argument("--classifier_style_group", required=True, type=str, choices=['All', 'No Action', 'Character'])


def get_cond_mode(args):
    # Allow explicit override first
    if hasattr(args, 'cond_mode') and args.cond_mode:
        return args.cond_mode
    if args.unconstrained:
        return 'no_cond'
    elif args.dataset in ['kit', 'humanml', '100style']:
        return 'text'
    else:
        return 'action'


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_lora_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_lora_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def edit_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    add_lora_options(parser)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)

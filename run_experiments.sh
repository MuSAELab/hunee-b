###############################################################################
# VAD Beehive Experiment
###############################################################################
for seed in 1234 5678 9012 3456 7890 6015 2017 1984 1776 2024;
do
    echo "================ Seed $seed ================"
    python run_downstream.py -m train -d vad_beehive -u fbank -n vad_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u fbank -n vad_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed

    python run_downstream.py -m train -d vad_beehive -u spectrogram -n vad_spec/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u spectrogram -n vad_spec/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed

    python run_downstream.py -m train -d vad_beehive -u mfcc -n vad_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u mfcc -n vad_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" --seed $seed

    python run_downstream.py -m train -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u byola -g upstream/byola/hparams/config.yaml -n vad_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u ssast -g upstream/ssast/hparams/config.yaml -n vad_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u ssast -g upstream/ssast/hparams/config.yaml -n vad_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u ssast -g upstream/ssast/hparams/config_small.yaml -n vad_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u ssast -g upstream/ssast/hparams/config_small.yaml -n vad_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u ssast -g upstream/ssast/hparams/config_base.yaml -n vad_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u ssast -g upstream/ssast/hparams/config_base.yaml -n vad_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u msm_mae -n vad_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u msm_mae -n vad_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u m2d -n vad_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u m2d -n vad_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed

    python run_downstream.py -m train -d vad_beehive -u beats -n vad_beats/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u beats -n vad_beats/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed

    python run_downstream.py -m train -d vad_beehive -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n vad_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
    python run_downstream.py -m evaluate -d vad_beehive -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n vad_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000/test" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
done


###############################################################################
# Buzz Identification
###############################################################################
for seed in 1234 5678 9012 3456 7890 6015 2017 1984 1776 2024;
do
    echo "================ Seed $seed ================"
    python run_downstream.py -m train -d buzz_identification -u fbank -n buzzid_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u fbank -n buzzid_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed

    python run_downstream.py -m train -d buzz_identification -u spectrogram -n buzzid_spec/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u spectrogram -n buzzid_spec/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed

    python run_downstream.py -m train -d buzz_identification -u mfcc -n buzzid_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u mfcc -n buzzid_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" --seed $seed

    python run_downstream.py -m train -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u byola -g upstream/byola/hparams/config.yaml -n buzzid_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u ssast -g upstream/ssast/hparams/config.yaml -n buzzid_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u ssast -g upstream/ssast/hparams/config.yaml -n buzzid_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u ssast -g upstream/ssast/hparams/config_small.yaml -n buzzid_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u ssast -g upstream/ssast/hparams/config_small.yaml -n buzzid_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u ssast -g upstream/ssast/hparams/config_base.yaml -n buzzid_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u ssast -g upstream/ssast/hparams/config_base.yaml -n buzzid_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u msm_mae -n buzzid_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u msm_mae -n buzzid_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u m2d -n buzzid_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u m2d -n buzzid_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed

    python run_downstream.py -m train -d buzz_identification -u beats -n buzzid_beats/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u beats -n buzzid_beats/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed

    python run_downstream.py -m train -d buzz_identification -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n buzzid_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
    python run_downstream.py -m evaluate -d buzz_identification -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n buzzid_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/BuzzDataset/processed" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
done

###############################################################################
# Beehive Strength
###############################################################################
for seed in 1234 5678 9012 3456 7890 6015 2017 1984 1776 2024;
do
    echo "================ Seed $seed ================"
    python run_downstream.py -m train -d beehive_strength -u fbank -n bee_str_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u fbank -n bee_str_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed

    python run_downstream.py -m train -d beehive_strength -u spectrogram -n bee_str_spec/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u spectrogram -n bee_str_spec/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed

    python run_downstream.py -m train -d beehive_strength -u mfcc -n bee_str_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u mfcc -n bee_str_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" --seed $seed

    python run_downstream.py -m train -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u byola -g upstream/byola/hparams/config.yaml -n beehive_str_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u ssast -g upstream/ssast/hparams/config.yaml -n beehive_str_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u ssast -g upstream/ssast/hparams/config.yaml -n beehive_str_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u ssast -g upstream/ssast/hparams/config_small.yaml -n beehive_str_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u ssast -g upstream/ssast/hparams/config_small.yaml -n beehive_str_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u ssast -g upstream/ssast/hparams/config_base.yaml -n beehive_str_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u ssast -g upstream/ssast/hparams/config_base.yaml -n beehive_str_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u msm_mae -n beehive_str_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u msm_mae -n beehive_str_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u m2d -n beehive_str_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u m2d -n beehive_str_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed

    python run_downstream.py -m train -d beehive_strength -u beats -n beehive_str_beats/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u beats -n beehive_str_beats/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed

    python run_downstream.py -m train -d beehive_strength -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n beehive_str_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
    python run_downstream.py -m evaluate -d beehive_strength -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n beehive_str_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/NectarDataset/beehive_strength/" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
done

###############################################################################
# Queenbee detection
###############################################################################
for seed in 1234 5678 9012 3456 7890 6015 2017 1984 1776 2024;
do
    echo "================ Seed $seed ================"
    python run_downstream.py -m train -d queenbee_detection -u fbank -n qb_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u fbank -n qb_fbank/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u spectrogram -n qb_spec/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u spectrogram -n qb_spec/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u mfcc -n qb_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u mfcc -n qb_mfcc/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/audioset/AudioNTT2020-BYOLA-64x96d2048.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola_b/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/finetuned/AudioNTT2020-BYOLA-64x96d2048b.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u byola -g upstream/byola/hparams/config.yaml -n qb_byola_ft/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/byola/AudioNTT2020-BYOLA-64x96d2048FT.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u ssast -g upstream/ssast/hparams/config.yaml -n qb_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u ssast -g upstream/ssast/hparams/config.yaml -n qb_ssast_tiny/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Tiny-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u ssast -g upstream/ssast/hparams/config_small.yaml -n qb_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u ssast -g upstream/ssast/hparams/config_small.yaml -n qb_ssast_small/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Small-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u ssast -g upstream/ssast/hparams/config_base.yaml -n qb_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u ssast -g upstream/ssast/hparams/config_base.yaml -n qb_ssast_base/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/ssast/SSAST-Base-Patch-400.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u msm_mae -n qb_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u msm_mae -n qb_msmmae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/msm_mae/80x512p16x16_0425/checkpoint-100.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u m2d -n qb_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u m2d -n qb_m2d/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/m2d/m2d_vit_base-80x608p16x16-220930-mr7/checkpoint-300.pth --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u beats -n qb_beats/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u beats -n qb_beats/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/beats/BEATs_iter3_plus_AS2M.pt --seed $seed

    python run_downstream.py -m train -d queenbee_detection -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n qb_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
    python run_downstream.py -m evaluate -d queenbee_detection -u cav_mae -g upstream/cav_mae/hparams/config.yaml -n qb_cav_mae/$seed -o"data_root=/media/heitor/Research/raw_files/nuhive/tasks/beehive_states_fold0-v2-full/16000" -k result/pretrain/cav_mae/cav_mae_scalepp.pth --seed $seed
done

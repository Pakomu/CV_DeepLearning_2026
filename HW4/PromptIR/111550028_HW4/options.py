import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=10, help='maximum number of epochs to train the total model.')
# ⚠️ 硬體優化：VRAM 8GB 建議設為 2，如果跑一半報錯 OOM (Out of Memory)，請改成 1
parser.add_argument('--batch_size', type=int, default=2, help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

# 保持 128，裁切夠小才能塞進 8GB VRAM
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
# ⚠️ 硬體優化：Windows 環境建議設為 2 或 4，避免 CPU 阻塞
parser.add_argument('--num_workers', type=int, default=4, help='number of workers.')

# path (保留原作者的，免得其他檔案報錯)
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/', help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/', help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/', help='where training images of dehazing saves.')

# ✨ 新增我們作業專用的資料夾路徑
parser.add_argument('--train_dir', type=str, default='./data/hw4_train/', help='我們自訂的 HW4 作業資料夾路徑')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/", help='checkpoint save path')

# ⚠️ 將預設改為 None，這樣程式就會乖乖使用本地端的 TensorBoard，不會逼你登入雲端
parser.add_argument("--wblogger", type=str, default=None, help="Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir", type=str, default="train_ckpt", help="Name of the Directory where the checkpoint is to be saved")
# ⚠️ 硬體優化：設定為 1 張顯示卡
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")

options = parser.parse_args()
def make_config(
    config_dir,
    config_file_name,
    super_loss_tau,
    super_loss_lam,
    super_loss_fac,
    dataset_name,
    dataset_folder,
):
    train_config_template = """
    "batch_size" : 32,
    "epochs" : 5,
	"sampling" : "closest",
    "super_loss": "true",
    "saving_path" : "./data/{}/results/",
    "name" : "{}",
    "data_path" : "./data/{}/",
    "model_name" : "multi-qa-distilbert-cos-v1",
    "neg_number" : 20,
    "partition_pattern":"internal",
    "alpha":0.1,
    "seed":47,
    "loss_alpha": 1.0,
    "super_loss_tau": {},
    "super_loss_lam": {},
    "super_loss_fac": {},
    "loss_beta": 0.0

""".format(
        dataset_folder,
        dataset_name,
        dataset_folder,
        super_loss_tau,
        super_loss_lam,
        super_loss_fac,
    )
    train_config_template = "{" + train_config_template + "}"

    with open(config_dir + config_file_name, "w") as f:
        f.write(train_config_template)


def make_experiment_script(
    experiment_file_name, config_dir, config_file_name, experiment_name
):
    experiment_script = """#!/bin/bash
#SBATCH --job-name=taxo
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=1800:00
#SBATCH --mem=100Gb

source ~/.bashrc
module load miniconda/3
module load cuda/11.1
conda create -n taxocomplete python=3.9
conda activate taxocomplete
/cvmfs/ai.mila.quebec/apps/x86_64/debian/miniconda/3/bin/python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c dglteam dgl-cuda11.1
pip install -r requirements
mkdir -p $SCRATCH/sentence-transformers
export SENTENCE_TRANSFORMERS_HOME=$SCRATCH/sentence-transformers
echo ""
echo ""
echo "{}:"
rm -rf $SLURM_TMPDIR/{}
mkdir -p $SLURM_TMPDIR/{}
find . -maxdepth 1 ! -name 'job_output.txt' ! -name 'job_error.txt' ! -name 'sbatch_out.txt' ! -name '.' ! -name 'cancel.sh' -exec mv {} $SLURM_TMPDIR/{}/ \;
cd $SLURM_TMPDIR/{}/
python ./src/train.py --config {}{}
echo "ERROR CODE: $?"
""".format(
        experiment_name,
        experiment_file_name,
        experiment_file_name,
        "{" + "}",
        experiment_file_name,
        experiment_file_name,
        config_dir,
        config_file_name,
    )

    with open("scripts/experiments/" + experiment_file_name, "w") as f:
        f.write(experiment_script)


datasets = [
    ("psychology", "MAG-PSY", "psy"),
    ("wordnet_verb", "SemEval-Verb", "semeval_verb"),
]

tau_values = [0.0]
lam_values = [0.01, 0.1, 1.0, 10]
fac_values = [0.01, 0.1, 0.5, 0.9, 1.0]

for dataset_name, dataset_folder, dataset_config_dir in datasets:
    for tau in tau_values:
        for lam in lam_values:
            for fac in fac_values:
                config_dir = "./config_files/{}/".format(dataset_config_dir)
                config_file_name = "superloss_t{}_l{}_f{}.json".format(tau, lam, fac)
                make_config(
                    config_dir,
                    config_file_name,
                    tau,
                    lam,
                    fac,
                    dataset_name,
                    dataset_folder,
                )
                experiment_file_name = "superloss_t{}_l{}_f{}_{}".format(
                    tau,
                    lam,
                    fac,
                    dataset_config_dir,
                )
                experiment_name = "SUPERLOSS TAU={} LAM={} FAC={} ({})".format(
                    tau,
                    lam,
                    fac,
                    dataset_config_dir.upper(),
                )
                make_experiment_script(
                    experiment_file_name,
                    config_dir,
                    config_file_name,
                    experiment_name,
                )

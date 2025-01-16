def make_config(
    config_dir,
    config_file_name,
    sampling_type,
    cossim_mapping_range_lower_bound,
    cossim_mapping_range_upper_bound,
    dataset_name,
    dataset_folder,
):
    train_config_template = """
    "batch_size" : 32,
    "epochs" : 5,
    "sampling" : "{}",
    "cossim_mapping_range": [{}, {}],
    "saving_path" : "./data/{}/results/",
    "name" : "{}",
    "data_path" : "./data/{}/",
    "model_name" : "multi-qa-distilbert-cos-v1",
    "neg_number" : 20,
    "partition_pattern":"internal",
    "alpha":0.1,
    "seed":47,
    "loss_alpha": 1.0,
    "loss_beta": 0.0

""".format(
        sampling_type,
        cossim_mapping_range_lower_bound,
        cossim_mapping_range_upper_bound,
        dataset_folder,
        dataset_name,
        dataset_folder,
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
echo ""
echo ""
echo "{}:"
rm -rf $SLURM_TMPDIR/{}
mkdir -p $SLURM_TMPDIR/{}
cp -r . $SLURM_TMPDIR/{}/
cd $SLURM_TMPDIR/{}/
python ./src/train.py --config {}{}
echo "ERROR CODE: $?"
""".format(
        experiment_name,
        experiment_file_name,
        experiment_file_name,
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

sampling_types = ["closest_range", "closest_range_linear"]

cosine_mapping_range_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for dataset_name, dataset_folder, dataset_config_dir in datasets:
    for sampling_type in sampling_types:
        for lower_bound in cosine_mapping_range_values:
            for upper_bound in cosine_mapping_range_values:
                if lower_bound < upper_bound:
                    config_dir = "./config_files/{}/".format(dataset_config_dir)
                    config_file_name = "{}_{}_{}.json".format(
                        "range" if sampling_type == "closest_range" else "linear",
                        lower_bound,
                        upper_bound,
                    )
                    make_config(
                        config_dir,
                        config_file_name,
                        sampling_type,
                        lower_bound,
                        upper_bound,
                        dataset_name,
                        dataset_folder,
                    )
                    experiment_file_name = "{}_{}_{}_{}".format(
                        "range" if sampling_type == "closest_range" else "linear",
                        lower_bound,
                        upper_bound,
                        dataset_config_dir,
                    )
                    experiment_name = "RANGE {} ({} to {}) {}".format(
                        "" if sampling_type == "closest_range" else "LINEAR",
                        lower_bound,
                        upper_bound,
                        dataset_config_dir.upper(),
                    )
                    make_experiment_script(
                        experiment_file_name,
                        config_dir,
                        config_file_name,
                        experiment_name,
                    )

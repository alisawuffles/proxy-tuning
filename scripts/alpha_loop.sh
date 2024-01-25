for a in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0
do
    id=$(sbatch --parsable --export=alpha=$a scripts/truthfulqa_alpha.sh)
    echo "$a: Submitted batch job $id"
done

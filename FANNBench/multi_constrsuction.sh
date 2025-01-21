echo "construct for different distribution"
# to ensure construction task can run in a whole day with the minmum guardance
# currently only for range query, which means label 

dataset_list=(
    sift10m
)

dist_list=(
    random
    in_dist
    out_dist
)

for dist in "${dist_list[@]}"; do
    echo "indexing for $dist"
    source all_construct.sh construction multi_cons $dist          # Acorn
done
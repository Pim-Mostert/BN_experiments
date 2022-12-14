registry="bd6edd4ab1f64cc6b843dd398eba3c02"
image="bd6edd4ab1f64cc6b843dd398eba3c02.azurecr.io/pim"
tag="4"

az acr build -t $image:$tag -r $registry .


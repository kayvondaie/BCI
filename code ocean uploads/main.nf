#!/usr/bin/env nextflow
// hash:sha256:578f992ac801c78ea220176b9cd5ffa283d63ccdb1c2e57c48100e23a6a466d5

nextflow.enable.dsl = 1

params.ophys_url = 's3://aind-private-data-prod-o5171v/single-plane-ophys_731012_2024-08-13_23-49-46'

ophys_to_aind_ophys_bergamo_stitcher_1 = channel.fromPath(params.ophys_url + "/", type: 'any')
ophys_to_aind_ophys_motion_correction_kd_2 = channel.fromPath(params.ophys_url + "/*.json", type: 'any')
capsule_aind_ophys_bergamo_stitcher_1_to_capsule_aind_ophys_motion_correction_kd_2_4 = channel.create()
ophys_to_aind_ophys_extraction_kd_6 = channel.fromPath(params.ophys_url + "/*.json", type: 'any')
capsule_aind_ophys_motion_correction_kd_2_to_capsule_aind_ophys_extraction_kd_3_7 = channel.create()

// capsule - aind-ophys-bergamo-stitcher
process capsule_aind_ophys_bergamo_stitcher_1 {
	tag 'capsule-6802467'
	container "$REGISTRY_HOST/published/bdd26885-5b39-46e3-a1a5-3c00f9f96add:v2"

	cpus 4
	memory '30 GB'

	input:
	path 'capsule/data' from ophys_to_aind_ophys_bergamo_stitcher_1.collect()

	output:
	path 'capsule/results/*' into capsule_aind_ophys_bergamo_stitcher_1_to_capsule_aind_ophys_motion_correction_kd_2_4

	script:
	"""
	#!/usr/bin/env bash
	set -e

	export CO_CAPSULE_ID=bdd26885-5b39-46e3-a1a5-3c00f9f96add
	export CO_CPUS=4
	export CO_MEMORY=32212254720

	mkdir -p capsule
	mkdir -p capsule/data && ln -s \$PWD/capsule/data /data
	mkdir -p capsule/results && ln -s \$PWD/capsule/results /results
	mkdir -p capsule/scratch && ln -s \$PWD/capsule/scratch /scratch

	echo "[${task.tag}] cloning git repo..."
	if [[ "\$(printf '%s\n' "2.20.0" "\$(git version | awk '{print \$3}')" | sort -V | head -n1)" = "2.20.0" ]]; then
		git -c credential.helper= clone --filter=tree:0 --branch v2.0 "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-6802467.git" capsule-repo
	else
		git -c credential.helper= clone --branch v2.0 "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-6802467.git" capsule-repo
	fi
	mv capsule-repo/code capsule/code && ln -s \$PWD/capsule/code /code
	rm -rf capsule-repo

	echo "[${task.tag}] running capsule..."
	cd capsule/code
	chmod +x run
	./run

	echo "[${task.tag}] completed!"
	"""
}

// capsule - aind-ophys-motion-correction-kd
process capsule_aind_ophys_motion_correction_kd_2 {
	tag 'capsule-1187361'
	container "$REGISTRY_HOST/capsule/2d8a5585-f1ee-4cbe-9b8b-fa6652e42c17:ecdf49b6d6102187f0c56e4aa00d6606"

	cpus 16
	memory '120 GB'

	publishDir "$RESULTS_PATH", saveAs: { filename -> new File(filename).getName() }

	input:
	path 'capsule/data/' from ophys_to_aind_ophys_motion_correction_kd_2.collect()
	path 'capsule/data/' from capsule_aind_ophys_bergamo_stitcher_1_to_capsule_aind_ophys_motion_correction_kd_2_4.collect()

	output:
	path 'capsule/results/*'
	path 'capsule/results/*' into capsule_aind_ophys_motion_correction_kd_2_to_capsule_aind_ophys_extraction_kd_3_7

	script:
	"""
	#!/usr/bin/env bash
	set -e

	export CO_CAPSULE_ID=2d8a5585-f1ee-4cbe-9b8b-fa6652e42c17
	export CO_CPUS=16
	export CO_MEMORY=128849018880

	mkdir -p capsule
	mkdir -p capsule/data && ln -s \$PWD/capsule/data /data
	mkdir -p capsule/results && ln -s \$PWD/capsule/results /results
	mkdir -p capsule/scratch && ln -s \$PWD/capsule/scratch /scratch

	echo "[${task.tag}] cloning git repo..."
	if [[ "\$(printf '%s\n' "2.20.0" "\$(git version | awk '{print \$3}')" | sort -V | head -n1)" = "2.20.0" ]]; then
		git -c credential.helper= clone --filter=tree:0 "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-1187361.git" capsule-repo
	else
		git -c credential.helper= clone "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-1187361.git" capsule-repo
	fi
	git -C capsule-repo checkout 14f07e43b4a3dcec0e774a26b6a05d432de0e711 --quiet
	mv capsule-repo/code capsule/code && ln -s \$PWD/capsule/code /code
	rm -rf capsule-repo

	echo "[${task.tag}] running capsule..."
	cd capsule/code
	chmod +x run
	./run

	echo "[${task.tag}] completed!"
	"""
}

// capsule - aind-ophys-extraction-kd
process capsule_aind_ophys_extraction_kd_3 {
	tag 'capsule-5369650'
	container "$REGISTRY_HOST/capsule/af640b43-bcf1-49e8-a0b9-138ad43bf560:fcfe05b841d123cc012a4bf40b987198"

	cpus 8
	memory '60 GB'

	publishDir "$RESULTS_PATH", saveAs: { filename -> new File(filename).getName() }

	input:
	path 'capsule/data/' from ophys_to_aind_ophys_extraction_kd_6.collect()
	path 'capsule/data/' from capsule_aind_ophys_motion_correction_kd_2_to_capsule_aind_ophys_extraction_kd_3_7

	output:
	path 'capsule/results/*'

	script:
	"""
	#!/usr/bin/env bash
	set -e

	export CO_CAPSULE_ID=af640b43-bcf1-49e8-a0b9-138ad43bf560
	export CO_CPUS=8
	export CO_MEMORY=64424509440

	mkdir -p capsule
	mkdir -p capsule/data && ln -s \$PWD/capsule/data /data
	mkdir -p capsule/results && ln -s \$PWD/capsule/results /results
	mkdir -p capsule/scratch && ln -s \$PWD/capsule/scratch /scratch

	echo "[${task.tag}] cloning git repo..."
	if [[ "\$(printf '%s\n' "2.20.0" "\$(git version | awk '{print \$3}')" | sort -V | head -n1)" = "2.20.0" ]]; then
		git -c credential.helper= clone --filter=tree:0 "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-5369650.git" capsule-repo
	else
		git -c credential.helper= clone "https://\$GIT_ACCESS_TOKEN@\$GIT_HOST/capsule-5369650.git" capsule-repo
	fi
	git -C capsule-repo checkout 7b825ad2df9d67f7f4a9c88a48106d1d73378ca5 --quiet
	mv capsule-repo/code capsule/code && ln -s \$PWD/capsule/code /code
	rm -rf capsule-repo

	echo "[${task.tag}] running capsule..."
	cd capsule/code
	chmod +x run
	./run

	echo "[${task.tag}] completed!"
	"""
}

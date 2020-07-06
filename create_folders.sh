for path in 'checkpoint' 'res/log' 'res/result' 'res/decode'; do
	mkdir -p $path/pretrain
	mkdir -p $path/finetune
done

rsync -ahv --exclude-from=.gitignore --exclude='.git/' ~/workspace/en_ja_translator_pytorch/ enigma21:~/work/en_ja_translator_pytorch
# # the sed invocation inserts the lines at the start of the file
# # after any initial comment lines
# sed -Ei -e '/^([^#]|$)/ {a \
# export PYENV_ROOT="$HOME/.pyenv"
# a \
# export PATH="$PYENV_ROOT/bin:$PATH"
# a \
# ' -e ':a' -e '$!{n;ba};}' ~/.profile
# echo 'eval "$(pyenv init --path)"' >>~/.profile

# echo 'eval "$(pyenv init -)"' >> ~/.bashrc
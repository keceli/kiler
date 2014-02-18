#-------------------
# Aliases (some from http://tldp.org/LDP/abs/html/sample-bashrc.html)
#-------------------
alias x=' exit'
alias cdu='cd ..'
alias cd2='cd ../..'
alias cd3='cd ../../../'
alias cd4='cd ../../../../'
alias cd5='cd ../../../../../'
alias cdb='cd -'
alias sdiffs='sdiff --suppress-common-lines'
alias viper='emacs -nw'


alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
# -> Prevents accidentally clobbering files.
alias mkdir='mkdir -p'

alias h='history'
alias j='jobs -l'
alias which='type -a'
alias ..='cd ..'

# Pretty-print of some PATH variables:
alias path='echo -e ${PATH//:/\\n}'
alias libpath='echo -e ${LD_LIBRARY_PATH//:/\\n}'


alias du='du -kh'    # Makes a more readable output.
alias df='df -kTh'

#-------------------------------------------------------------
# The 'ls' family (this assumes you use a recent GNU ls).
#-------------------------------------------------------------
if ls --color=auto / >/dev/null 2>&1
then
 alias ls='ls -hC --color=auto' # GNU ls
 alias ll="ls -lvt --group-directories-first"
else
 alias ls='ls -hC -G' # BSD ls
 alias ll="ls -lvt "
fi
#alias ls='ls -h --color'
alias lx='ls -lXB'         #  Sort by extension.
alias lk='ls -lS'         #  Sort by size
alias lt='ls -lt'         #  Sort by date
alias lc='ls -ltc'        #  Sort by/show change time
alias lu='ls -ltu'        #  Sort by/show access time

# The ubiquitous 'll': directories first, with alphanumeric sorting:
alias lm='ll |more'        #  Pipe through 'more'
alias lr='ll -R'           #  Recursive ls.
alias la='ll -A'           #  Show hidden files.
alias tree='tree -Csuh'    #  Nice alternative to 'recursive ls' ...

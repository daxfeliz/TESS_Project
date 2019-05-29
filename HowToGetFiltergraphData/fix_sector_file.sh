

for oldname in *txt
do
        newname=`echo $oldname | sed -e 's/ /_/g'`
        mv "$oldname" temp.txt
        sed '1d' temp.txt > "$newname"
done

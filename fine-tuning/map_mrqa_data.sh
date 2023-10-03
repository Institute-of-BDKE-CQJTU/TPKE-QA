for item in {bioasq,squad,hotpotqa,naturalquestions,newsqa,textbookqa,searchqa,triviaqa}; do
{
    python fine-tuning/delete_special_symbol.py --dataset=${item} &
}
done
wait
echo "finish"

python fine-tuning/qass_preprocess.py --path "./data/mrqa-few-shot/*/*_delete_special_symbol.jsonl"
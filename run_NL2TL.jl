using PyCall
#using Spot


# import packages from python
torch = pyimport("torch")
tr = pyimport("transformers")
AutoTokenizer = tr.AutoTokenizer
AutoModelForSeq2SeqLM = tr.AutoModelForSeq2SeqLM
T5Tokenizer = tr.T5Tokenizer


pd = pyimport("pandas")
datasets = pyimport("datasets")
py_Dataset = datasets.Dataset
py_DatasetDict = datasets.DatasetDict
py_load_dataset = datasets.load_dataset
py_load_from_disk = datasets.load_from_disk
tqdm = pyimport("tqdm")
sys = pyimport("sys")
os = pyimport("os")
argparse = pyimport("argparse")
ipy_error = pyimport("IPython.core.error")
random = pyimport("random")
np = pyimport("numpy")
nltk = pyimport("nltk")
json = pyimport("json")
cvs = pyimport("csv")


output_dir = "/Users/abayat/Dio/Dionysos.jl_1/NL2TL/checkpoint-62500"


model_checkpoint = "t5-base"
prefix = "Transform the following sentence into Signal Temporal logic: "




max_input_length = 1024
max_target_length = 128



# -----------------------------------------------------------------------------
# Load tokenizer/model robustly.
# Your checkpoint folder currently contains only T5 weights (pytorch_model.bin)
# and does NOT contain HuggingFace metadata (config.json / tokenizer files).
# So we:
#   1) load tokenizer + base T5 model from `model_checkpoint`
#   2) load fine-tuned weights from `output_dir/pytorch_model.bin`
# -----------------------------------------------------------------------------

# Always load tokenizer from the base checkpoint (it ships the tokenizer files)
# NOTE: you can switch to `T5TokenizerFast` explicitly if you prefer.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=max_input_length)

# Device
device = torch.device(torch.cuda.is_available() ? "cuda:0" : "cpu")

# Build a T5 model skeleton from the base config, then load fine-tuned weights
T5ForConditionalGeneration = tr.T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

# Load your fine-tuned state dict (works when output_dir only has pytorch_model.bin)
weights_path = joinpath(output_dir, "pytorch_model.bin")
@assert isfile(weights_path) "Expected fine-tuned weights at: $(weights_path)"
state = torch.load(weights_path; map_location="cpu")
missing = model.load_state_dict(state; strict=false)

# Sanity print (optional)
# println("Loaded fine-tuned weights. Missing/unexpected keys: ", missing)



# Write NL using your map vocabulary (blue/brown/purple/green/yellow/obstacle)
# This will be converted to prop_i for the LLM automatically.
####################################################################################################
############################################# NL Input #############################################
####################################################################################################
#NL_sentence = "Head first towards prop_1, then proceed to prop_2. Don't miss out on prop_3. Round off your trek in prop_4."
# NL_sentence = "(blue) or (brown) never happens"
# NL_sentence = "(blue) or never (brown)"
# NL_sentence = "Globally, everytime when (brown) and (blue) then all of the following conditions are true : (green)."
# NL_sentence = "Globally, everytime when (brown) and (blue) then (green) is true."
NL_sentence = "Either prop_1 holds, or from the next step onward, prop_2 remains true forever."
####################################################################################################

println("NL (raw): ", NL_sentence)

inputs = [prefix * NL_sentence]
inputs = tokenizer(inputs; max_length=max_input_length, truncation=true, return_tensors="pt").to(device)
output = model.generate(
    input_ids = inputs["input_ids"],
    attention_mask = inputs["attention_mask"],
    num_beams = 8,
    do_sample = true,
    max_length = max_target_length,
)


println(output)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=true)[1]
println(decoded_output)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=true)[1]

println(decoded_output)


println("LLM output: ", decoded_output)

function llm_to_spot_ltl(s::AbstractString)::String
   # 1) normalize whitespace
   t = strip(replace(String(s), r"\s+" => " "))


   # 2) tokenize words and punctuation we care about
   # tokens: identifiers (prop_1), operators, parentheses
   toks = String[]
   i = 1
   while i <= lastindex(t)
       c = t[i]
       if c in ('(', ')')
           push!(toks, string(c))
           i = nextind(t, i)
       elseif c in ('&','|','!')
           push!(toks, string(c))
           i = nextind(t, i)
       elseif c == '-'
           # possibly "->"
           if i < lastindex(t) && t[nextind(t,i)] == '>'
               push!(toks, "->")
               i = nextind(t, nextind(t,i))
           else
               push!(toks, "-")
               i = nextind(t,i)
           end
       elseif isspace(c)
           i = nextind(t, i)
       else
           # read a word/identifier
           j = i
           while j <= lastindex(t)
               cj = t[j]
               if isspace(cj) || cj in ('(', ')', '&', '|', '!', '-')
                   break
               end
               j = nextind(t, j)
           end
           push!(toks, t[i:prevind(t,j)])
           i = j
       end
   end


   # 3) map English-like keywords to Spot tokens
   function map_token(tok::String)::String
       w = lowercase(tok)
       if w == "globally" || w == "always"
           return "G"
       elseif w == "finally" || w == "eventually"
           return "F"
       elseif w == "next"
           return "X"
       elseif w == "until"
           return "U"
       elseif w == "release" || w == "releases"
           return "R"
       elseif w == "and"
           return "&"
       elseif w == "or"
           return "|"
       elseif w == "not" || w == "negation" || w == "negate"
           return "!"
       elseif w == "imply" || w == "implies" || w == "implie" || w == "implicate" || w == "implies," || w == "imply,"
           return "->"
       elseif w == "true" || w == "tt"
           return "true"
       elseif w == "false" || w == "ff"
           return "false"
       else
           return tok
       end
   end


   toks = [map_token(tok) for tok in toks if !isempty(tok)]


   # 4) rebuild string with sensible spacing:
   #    - no spaces after unary temporal ops before '(' when present: G( ... )
   #    - otherwise keep single spaces between tokens
   out = IOBuffer()
   prev = ""
   for tok in toks
       if tok == ")"
           print(out, tok)
       elseif tok == "("
           # attach "G(" and "F(" etc.
           if prev in ("G","F","X")
               # remove trailing space if any
               str = String(take!(out))
               str = replace(str, r"\s+$" => "")
               out = IOBuffer(); print(out, str)
               print(out, tok)
           else
               if position(out) > 0 && !endswith(String(take!(IOBuffer(String(take!(out))))), " ")
                   # no-op; we handle spacing below
               end
               # restore out
               # (simpler: ensure a space if needed)
               #
               # We'll just add directly with a preceding space if last char is alnum or ')'
               str = String(take!(out))
               out = IOBuffer(); print(out, str)
               if !isempty(str)
                   lastc = str[end]
                   if isletter(lastc) || isnumeric(lastc) || lastc == ')'
                       print(out, ' ')
                   end
               end
               print(out, tok)
           end
       else
           # normal token
           str = String(take!(out))
           out = IOBuffer(); print(out, str)
           if !isempty(str)
               lastc = str[end]
               if lastc != '(' && lastc != ' '
                   print(out, ' ')
               end
           end
           print(out, tok)
       end
       prev = tok
   end


   res = strip(String(take!(out)))


   # 5) fix common LLM artifact: extra closing parentheses
   n_open  = count(==( '(' ), res)
   n_close = count(==( ')' ), res)
   if n_close > n_open
       # remove extra ')' from the end only
       extra = n_close - n_open
       while extra > 0 && endswith(res, ")")
           res = chop(res)
           extra -= 1
           res = rstrip(res)
       end
   end


   # 6) final whitespace cleanup
   res = replace(res, r"\s+" => " ")
   return res
end


spot_ltl = llm_to_spot_ltl(decoded_output)
println("LLM output (Spot LTL): ", spot_ltl)
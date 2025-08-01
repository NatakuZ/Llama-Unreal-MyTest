// Copyright 2025-current Getnamo.

#include "Internal/LlamaInternal.h"
#include "common/common.h"
#include "common/sampling.h"
#include "LlamaDataTypes.h"
#include "LlamaUtility.h"
#include "HardwareInfo.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Embedding/VectorDatabase.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Async/ParallelFor.h"

#include "JsonObjectConverter.h"

static void MyLlamaLogCallback(ggml_log_level level, const char* text, void* user_data)
{
    // Converte il livello di log di ggml in quello di Unreal
    ELogVerbosity::Type Verbosity = ELogVerbosity::Log;
    switch (level)
    {
    case GGML_LOG_LEVEL_ERROR:
        Verbosity = ELogVerbosity::Error;
        break;
    case GGML_LOG_LEVEL_WARN:
        Verbosity = ELogVerbosity::Warning;
        break;
    default:
        Verbosity = ELogVerbosity::Log;
        break;
    }

    // Pulisce il messaggio da eventuali "a capo" finali
    FString Message = FString(UTF8_TO_TCHAR(text));
    Message.TrimEndInline();

    // Stampa il messaggio nell'Output Log di Unreal
 //   UE_LOG(LlamaLog, Verbosity, TEXT("[llama.cpp] %s"), *Message);
}

bool FLlamaInternal::LoadModelFromParams(const FLLMModelParams& InModelParams)
{
    llama_log_set(MyLlamaLogCallback, nullptr);
       ggml_backend_load_all();
    FString RHI = FHardwareInfo::GetHardwareDetailsString();
    FString GPU = FPlatformMisc::GetPrimaryGPUBrand();

    UE_LOG(LogTemp, Log, TEXT("Device Found: %s %s"), *GPU, *RHI);

    LastLoadedParams = InModelParams;

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */)
        {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    std::string ModelPath = TCHAR_TO_UTF8(*FLlamaPaths::ParsePathIntoFullPath(InModelParams.PathToModel));


    //CommonParams Init (false)//
    if (false)
        //if (InModelParams.Advanced.bUseCommonParams || InModelParams.Advanced.bEmbeddingMode)
    {
        //use common init
        common_init();

        common_params CommonParams;
        CommonParams.n_ctx = InModelParams.MaxContextLength;
        CommonParams.n_batch = InModelParams.MaxBatchLength;
        CommonParams.cpuparams.n_threads = InModelParams.Threads;
        CommonParams.embedding = InModelParams.Advanced.bEmbeddingMode;  //true
        CommonParams.n_gpu_layers = InModelParams.GPULayers;
        CommonParams.model.path = ModelPath;

        common_init_result LlamaInit = common_init_from_params(CommonParams);

        LlamaModel = LlamaInit.model.get();
        Context = LlamaInit.context.get();

        //Sanity check the model settings for embedding
        if (CommonParams.embedding)
        {
            if (llama_model_has_encoder(LlamaModel) && llama_model_has_decoder(LlamaModel))
            {
                EmitErrorMessage(TEXT("computing embeddings in encoder-decoder models is not supported"), 41, __func__);
                return false;
            }

            const int n_ctx_train = llama_model_n_ctx_train(LlamaModel);
            const int n_ctx = llama_n_ctx(Context);

            if (n_ctx > n_ctx_train)
            {
                FString ErrorMessage = FString::Printf(TEXT("warning: model was trained on only % d context tokens(% d specified)"), n_ctx_train, n_ctx);
                EmitErrorMessage(ErrorMessage, 42, __func__);
                return false;
            }
        }
    }
    else
    {
        //Regular init
        // initialize the model
        llama_model_params LlamaModelParams = llama_model_default_params();
        LlamaModelParams.n_gpu_layers = InModelParams.GPULayers;

        LlamaModel = llama_model_load_from_file(ModelPath.c_str(), LlamaModelParams);
        if (!LlamaModel)
        {
            FString ErrorMessage = FString::Printf(TEXT("Unable to load model at <%hs>"), ModelPath.c_str());
            EmitErrorMessage(ErrorMessage, 10, __func__);
            return false;
        }

        llama_context_params ContextParams = llama_context_default_params();
        ContextParams.n_ctx = InModelParams.MaxContextLength;
        ContextParams.n_batch = InModelParams.MaxBatchLength;
        ContextParams.n_threads = InModelParams.Threads;
        ContextParams.n_threads_batch = InModelParams.Threads;

        //only set if true
        if (InModelParams.Advanced.bEmbeddingMode)
        {
            ContextParams.embeddings = InModelParams.Advanced.bEmbeddingMode;  //to be tested for A/B comparison if it works
            switch (PoolingTypeOfModel) {
            case -1:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
                break;
            case 0:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_NONE;
                break;
            case 1:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_MEAN;
                break;
            case 2:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_CLS;
                break;
            case 3:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_LAST;
                break;
            case 4:
                ContextParams.pooling_type = LLAMA_POOLING_TYPE_RANK;
                break;
            }
        }

        Context = llama_init_from_model(LlamaModel, ContextParams);
    }

    if (!Context)
    {
        FString ErrorMessage = FString::Printf(TEXT("Unable to initialize model with given context params."));
        EmitErrorMessage(ErrorMessage, 11, __func__);
        return false;
    }

    //Only standard mode uses sampling
    if (!InModelParams.Advanced.bEmbeddingMode)
    {
        //common sampler strategy
        if (InModelParams.Advanced.bUseCommonSampler)
        {
            common_params_sampling SamplingParams;

            if (InModelParams.Advanced.MinP != -1.f)
            {
                SamplingParams.min_p = InModelParams.Advanced.MinP;
            }
            if (InModelParams.Advanced.TopK != -1.f)
            {
                SamplingParams.top_k = InModelParams.Advanced.TopK;
            }
            if (InModelParams.Advanced.TopP != -1.f)
            {
                SamplingParams.top_p = InModelParams.Advanced.TopP;
            }
            if (InModelParams.Advanced.TypicalP != -1.f)
            {
                SamplingParams.typ_p = InModelParams.Advanced.TypicalP;
            }
            if (InModelParams.Advanced.Mirostat != -1)
            {
                SamplingParams.mirostat = InModelParams.Advanced.Mirostat;
                SamplingParams.mirostat_eta = InModelParams.Advanced.MirostatEta;
                SamplingParams.mirostat_tau = InModelParams.Advanced.MirostatTau;
            }

            //Seed is either default or the one specifically passed in for deterministic results
            if (InModelParams.Seed != -1)
            {
                SamplingParams.seed = InModelParams.Seed;
            }

            CommonSampler = common_sampler_init(LlamaModel, SamplingParams);
        }

        Sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

        //Temperature is always applied
        llama_sampler_chain_add(Sampler, llama_sampler_init_temp(InModelParams.Advanced.Temp));

        //If any of the repeat penalties are set, apply penalties to sampler
        if (InModelParams.Advanced.PenaltyLastN != 0 ||
            InModelParams.Advanced.PenaltyRepeat != 1.f ||
            InModelParams.Advanced.PenaltyFrequency != 0.f ||
            InModelParams.Advanced.PenaltyPresence != 0.f)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_penalties(
                InModelParams.Advanced.PenaltyLastN, InModelParams.Advanced.PenaltyRepeat,
                InModelParams.Advanced.PenaltyFrequency, InModelParams.Advanced.PenaltyPresence));
        }

        //Optional sampling strategies - MinP should be applied by default of 0.05f
        if (InModelParams.Advanced.MinP != -1.f)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_min_p(InModelParams.Advanced.MinP, 1));
        }
        if (InModelParams.Advanced.TopK != -1.f)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_top_k(InModelParams.Advanced.TopK));
        }
        if (InModelParams.Advanced.TopP != -1.f)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_top_p(InModelParams.Advanced.TopP, 1));
        }
        if (InModelParams.Advanced.TypicalP != -1.f)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_typical(InModelParams.Advanced.TypicalP, 1));
        }
        if (InModelParams.Advanced.Mirostat != -1)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_mirostat_v2(
                InModelParams.Advanced.Mirostat, InModelParams.Advanced.MirostatTau, InModelParams.Advanced.MirostatEta));
        }

        //Seed is either default or the one specifically passed in for deterministic results
        if (InModelParams.Seed == -1)
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        }
        else
        {
            llama_sampler_chain_add(Sampler, llama_sampler_init_dist(InModelParams.Seed));
        }

        //NB: this is just a starting heuristic, 
        ContextHistory.reserve(1024);

    }//End non-embedding mode

    //empty by default
    Template = std::string();
    TemplateSource = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);

    //Prioritize: custom jinja, then name, then default
    if (!InModelParams.CustomChatTemplate.Jinja.IsEmpty())
    {
        Template = FLlamaString::ToStd(InModelParams.CustomChatTemplate.Jinja);
        if (InModelParams.CustomChatTemplate.TemplateSource.IsEmpty())
        {
            TemplateSource = std::string("Custom Jinja");
        }
    }
    else if (!InModelParams.CustomChatTemplate.TemplateSource.IsEmpty() &&
        InModelParams.CustomChatTemplate.TemplateSource != TEXT("tokenizer.chat_template"))
    {
        //apply template source name, this may fail
        std::string TemplateName = FLlamaString::ToStd(InModelParams.CustomChatTemplate.TemplateSource);
        const char* TemplatePtr = llama_model_chat_template(LlamaModel, TemplateName.c_str());

        if (TemplatePtr != nullptr)
        {
            Template = std::string(TemplatePtr);
        }
    }

    if (InModelParams.Advanced.bEmbeddingMode)
    {
        Template = std::string("");
        TemplateSource = std::string("embedding mode, templates not used");
    }
    else
    {

        if (Template.empty())
        {
            const char* TemplatePtr = llama_model_chat_template(LlamaModel, nullptr);

            if (TemplatePtr != nullptr)
            {
                Template = std::string(TemplatePtr);
                TemplateSource = std::string("tokenizer.chat_template");
            }
        }
    }

    FilledContextCharLength = 0;

    bIsModelLoaded = true;

    return true;
}

void FLlamaInternal::UnloadModel()
{
    if (Sampler)
    {
        llama_sampler_free(Sampler);
        Sampler = nullptr;
    }
    if (Context)
    {
        llama_free(Context);
        Context = nullptr;
    }
    if (LlamaModel)
    {
        llama_model_free(LlamaModel);
        LlamaModel = nullptr;
    }
    if (CommonSampler)
    {
        common_sampler_free(CommonSampler);
        CommonSampler = nullptr;
    }

    ContextHistory.clear();

    bIsModelLoaded = false;
}

std::string FLlamaInternal::WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS)
{
    std::vector<llama_chat_message> MessageListWrapper;
    MessageListWrapper.push_back({ RoleForEnum(Role), _strdup(Text.c_str()) });

    //pre-allocate buExample3er 2x the size of text
    std::vector<char> BuExample3er;

    int32 NewLen = 0;

    if (OverrideTemplate.empty())
    {
        NewLen = ApplyTemplateFromMessagesToBuExample3er(Template, MessageListWrapper, BuExample3er, bAddAssistantBoS);
    }
    else
    {
        NewLen = ApplyTemplateFromMessagesToBuExample3er(OverrideTemplate, MessageListWrapper, BuExample3er, bAddAssistantBoS);
    }

    if (NewLen > 0)
    {
        return std::string(BuExample3er.data(), BuExample3er.data() + NewLen);
    }
    else
    {
        return std::string("");
    }
}

void FLlamaInternal::StopGeneration()
{
    bGenerationActive = false;
}

bool FLlamaInternal::IsGenerating()
{
    return bGenerationActive;
}

int32 FLlamaInternal::MaxContext()
{
    if (Context)
    {
        return llama_n_ctx(Context);
    }
    else
    {
        return 0;
    }
}

int32 FLlamaInternal::UsedContext()
{
    if (Context)
    {
        return llama_kv_self_used_cells(Context);
    }
    else
    {
        return 0;
    }
}

bool FLlamaInternal::IsModelLoaded()
{
    return bIsModelLoaded;
}

void FLlamaInternal::ResetContextHistory(bool bKeepSystemsPrompt)
{
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (bKeepSystemsPrompt)
    {
        //Valid trim case
        if (Messages.size() > 1)
        {  // per ora non lo usiamo 
            //Rollback all the messages except the first one
            RollbackContextHistoryByMessages(Messages.size() - 1);
            return;
        }
    }
    /* old version 
    //Full Reset
    ContextHistory.clear();
    Messages.clear();

    llama_kv_self_clear(Context);
    FilledContextCharLength = 0;
    */
    //Full Reset
    ContextHistory.clear();
    Messages.clear();

    // MODIFICA: Usa la funzione standard e più completa per pulire la cache.
    llama_kv_cache_clear(Context);

    FilledContextCharLength = 0;
}

void FLlamaInternal::RollbackContextHistoryByTokens(int32 NTokensToErase)
{
    // clear the last n_regen tokens from the KV cache and update n_past
    int32 TokensUsed = llama_kv_self_used_cells(Context); //FilledContextCharLength

    llama_kv_self_seq_rm(Context, 0, TokensUsed - NTokensToErase, -1);

    //FilledContextCharLength -= NTokensToErase;

    //Run a decode to sync everything else
    //llama_decode(Context, llama_batch_get_one(nullptr, 0));
}

void FLlamaInternal::RollbackContextHistoryByMessages(int32 NMessagesToErase)
{
    //cannot do rollback if model isn't loaded, ignore.
    if (!bIsModelLoaded)
    {
        return;
    }

    if (IsGenerating())
    {
        StopGeneration();
    }

    if (NMessagesToErase <= Messages.size())
    {
        Messages.resize(Messages.size() - NMessagesToErase);
    }

    //Obtain full prompt before it gets deleted
    std::string FullPrompt(ContextHistory.data(), ContextHistory.data() + FilledContextCharLength);

    //resize the context history
    int32 NewLen = ApplyTemplateToContextHistory(false);

    //tokenize to find out how many tokens we need to remove

    //Obtain new prompt, find delta
    std::string FormattedPrompt(ContextHistory.data(), ContextHistory.data() + NewLen);

    std::string PromptToRemove(FullPrompt.substr(FormattedPrompt.length()));

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const int NPromptTokens = -llama_tokenize(Vocab, PromptToRemove.c_str(), PromptToRemove.size(), NULL, 0, false, true);

    //now rollback KV-cache
    RollbackContextHistoryByTokens(NPromptTokens);

    //Sync resized length;
    FilledContextCharLength = NewLen;

    //Shrink to fit
    ContextHistory.resize(FilledContextCharLength);
}

std::string FLlamaInternal::InsertRawPrompt(const std::string& Prompt, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return 0;
    }

    int32 TokensProcessed = ProcessPrompt(Prompt);

    FLlamaString::AppendToCharVector(ContextHistory, Prompt);

    if (bGenerateReply)
    {
        std::string Response = Generate("", false);
        FLlamaString::AppendToCharVector(ContextHistory, Response);
    }
    return "";
}

std::string FLlamaInternal::InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role, bool bAddAssistantBoS, bool bGenerateReply)
{
    if (!bIsModelLoaded)
    {
        UE_LOG(LlamaLog, Warning, TEXT("Model isn't loaded"));
        return std::string();
    }

    int32 NewLen = FilledContextCharLength;

    if (!Prompt.empty())
    {
        Messages.push_back({ RoleForEnum(Role), _strdup(Prompt.c_str()) });

        NewLen = ApplyTemplateToContextHistory(bAddAssistantBoS);
    }

    std::string FormattedPrompt(ContextHistory.data() + FilledContextCharLength, ContextHistory.data() + NewLen);

    int32 TokensProcessed = ProcessPrompt(FormattedPrompt, Role);

    FilledContextCharLength = NewLen;

    //Check for a reply if we want to generate one, otherwise return an empty reply
    std::string Response;
    if (bGenerateReply)
    {
        //Run generation
        Response = Generate();
    }
    UE_LOG(LlamaLog, Warning, TEXT("Response generate"));
    return Response;
}



void FLlamaInternal::InsertSentencesInEmbeddedModel(TArray<FString> Sentences)
{/*
    for (const std::string& sentence_text : input_sentences) {
        // 1. Tokenizzazione
        // Pre-allocare un buExample3er per i token. ctx_params.n_ctx è una stima sicura.
        Context
        std::vector<llama_token> tokens();
    }
    */
}

std::string FLlamaInternal::ResumeGeneration()
{
    //Todo: erase last assistant message to merge the two messages if the last message was the assistant one.

    //run an empty user prompt
    return Generate();
}
//numero dei chunck totale 
int32 n_chunks = 0;
const float* chunks;
void FLlamaInternal::GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings)
{
    //apply https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp wrapping logic

    if (!Context)
    {
        EmitErrorMessage(TEXT("Context invalid, did you load the model?"), 43, __func__);
        return;
    }

    //Tokenize prompt - we're crashing out here... 
    //Check if our sampling/etc params are wrong or vocab is wrong.
    //Try tokenizing using normal method?
    //CONTINUE HERE:
    //FString TextStr(Text.c_str());
    //  TArray<FString> Parsed;
    // TextStr.ParseIntoArray(Parsed,TEXT("/n"));
    // UE_LOG(LogTemp, Log, TEXT("Trying to sample <%hs>"), Text.c_str());
    // std::vector<chunk> chunks;
    auto Input = common_tokenize(Context, Text, true, true);

    //int32 NBatch = llama_n_ctx(Context);    //todo: get this from our params
    int32 NBatch = Input.size();    //todo: get this from our params

    llama_batch Batch = llama_batch_init(NBatch, 0, 1);
    //llama_batch Batch = llama_batch_get_one(Input.data(), Input.size());

    //add single batch
    BatchAddSeq(Batch, Input, 0);

    enum llama_pooling_type PoolingType = llama_pooling_type(Context);

    //Count number of embeddings
    int32 EmbeddingCount = 0;

    if (PoolingType == llama_pooling_type::LLAMA_POOLING_TYPE_NONE)
    {
        EmbeddingCount = 1;// Input.size();
    }
    else
    {
        EmbeddingCount = 1;
    }
    n_chunks= EmbeddingCount;
    int32 NEmbd = llama_model_n_embd(LlamaModel);
   
    Embeddings = std::vector<float>(EmbeddingCount * NEmbd, 0);

    EmbeddingsPtr = Embeddings.data();

    // std::vector<chunk> chunks;
    /*for (int i = 0; i < Embeddings.size(); i++)
    {
        Chunk chunk;
        chunk.Text = "";
    //    chunk.Embedding = EmbeddingsPtr[i];

    }*/
    //decode
    BatchDecodeEmbedding(Context, Batch, EmbeddingsPtr, 0, NEmbd, 2);

    //if (PoolingTypeOfModel == 0)
    //{
    ////type none
    //    for (int j = 0; j < NBatch; j++) 
    //    {
    //        UE_LOG(LogTemp, Log, TEXT("embedding %d: "),j);
    //        for (int i = 0; i < std::min(3, NEmbd); i++) 
    //        {
    //            //the value is defualt 2 
    //            /* if (params.embd_normalize == 0) {
    //                     UE_LOG(LogTemp, Log, TEXT("%6.0f ", EmbeddingsPtr[j * NEmbd + i]));
    //            }
    //            else {*/
    //            UE_LOG(LogTemp, Log, TEXT("%9.6f "), EmbeddingsPtr[j * NEmbd + i]);
    //            
    //        }
    //        UE_LOG(LogTemp, Log, TEXT("..."));
    //        for (int i = NEmbd - 3; i < NEmbd; i++) 
    //        {
    //           /* if (params.embd_normalize == 0) {
    //                UE_LOG(LogTemp, Log, TEXT("%6.0f ", EmbeddingsPtr[j * NEmbd + i]));
    //            }
    //            else {*/
    //            UE_LOG(LogTemp, Log, TEXT("%9.6f "), EmbeddingsPtr[j * NEmbd + i]);
    //            
    //        }
    //    }
    //}
   


}

int32 FLlamaInternal::ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role)
{
  //  UE_LOG(LlamaLog, Log, TEXT("prompt pre process  %s "), Prompt);
   //FString PromptFstring(Prompt.c_str());
   // UE_LOG(LlamaLog, Log, TEXT("prompt pre proces %s "), *PromptFstring);
    const auto StartTime = ggml_time_us();

    //Grab vocab
    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);
    const bool IsFirst = llama_kv_self_used_cells(Context) == 0;

    // tokenize the prompt
    const int NPromptTokens = -llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), NULL, 0, IsFirst, true);
    std::vector<llama_token> PromptTokens(NPromptTokens);
    if (llama_tokenize(Vocab, Prompt.c_str(), Prompt.size(), PromptTokens.data(), PromptTokens.size(), IsFirst, true) < 0)
    {
        EmitErrorMessage(TEXT("failed to tokenize the prompt"), 21, __func__);
        return NPromptTokens;
    }

    //All in one batch
    if (LastLoadedParams.Advanced.PromptProcessingPacingSleep == 0.f)
    {
        // prepare a batch for the prompt
        llama_batch Batch = llama_batch_get_one(PromptTokens.data(), PromptTokens.size());
        
        //check sizing before running prompt decode
        int NContext = llama_n_ctx(Context);
        int NContextUsed = llama_kv_self_used_cells(Context);

        if (NContextUsed + NPromptTokens > NContext)
        {
            ResponseStatus = ELLMResponseStatus::ContextOversize; 
            EmitErrorMessage(FString::Printf(
                TEXT("Failed to insert, tried to insert %d tokens to currently used %d tokens which is more than the max %d context size. Try increasing the context size and re-run prompt."),
                NPromptTokens, NContextUsed, NContext
            ), 22, __func__);
            return 0;
        }
        else
        {
			UE_LOG(LlamaLog, Log, TEXT("Context is valid, used %d tokens out of %d"), NContextUsed, NContext);
        }

        // run it through the decode (input)
        if (llama_decode(Context, Batch))
        {
			UE_LOG(LlamaLog, Warning, TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."));
            EmitErrorMessage(TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."), 23, __func__);
            return NPromptTokens;
        }
    }
    //Split it and sleep between batches for pacing purposes
    else
    {
        int32 BatchCount = LastLoadedParams.Advanced.PromptProcessingPacingSplitN;

        int32 TotalTokens = PromptTokens.size();
        int32 TokensPerBatch = TotalTokens / BatchCount;
        int32 Remainder = TotalTokens % BatchCount;

        int32 StartIndex = 0;

        for (int32 i = 0; i < BatchCount; i++)
        {
            // Calculate how many tokens to put in this batch
            int32 CurrentBatchSize = TokensPerBatch + (i < Remainder ? 1 : 0);

            // Slice the relevant tokens for this batch
            std::vector<llama_token> BatchTokens(
                PromptTokens.begin() + StartIndex,
                PromptTokens.begin() + StartIndex + CurrentBatchSize
            );

            // Prepare the batch
            llama_batch Batch = llama_batch_get_one(BatchTokens.data(), BatchTokens.size());

            // Check context before running decode
            int NContext = llama_n_ctx(Context);
            int NContextUsed = llama_kv_self_used_cells(Context);

            if (NContextUsed + BatchTokens.size() > NContext)
            {
                ResponseStatus = ELLMResponseStatus::ContextOversize;
                EmitErrorMessage(FString::Printf(
                    TEXT("Failed to insert, tried to insert %d tokens to currently used %d tokens which is more than the max %d context size. Try increasing the context size and re-run prompt."),
                    BatchTokens.size(), NContextUsed, NContext
                ), 22, __func__);
                return 0;
            }

            // Decode this batch
            if (llama_decode(Context, Batch))
            {
                EmitErrorMessage(TEXT("Failed to decode, could not find a KV slot for the batch (try reducing the size of the batch or increase the context)."), 23, __func__);
                return BatchTokens.size();
            }

            StartIndex += CurrentBatchSize;
            FPlatformProcess::Sleep(LastLoadedParams.Advanced.PromptProcessingPacingSleep);
        }
    }

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (OnPromptProcessed)
    {
        float Speed = NPromptTokens / Duration;
        OnPromptProcessed(NPromptTokens, Role, Speed);
    }

    return NPromptTokens;
}

std::string FLlamaInternal::Generate(const std::string& Prompt, bool bAppendToMessageHistory)
{
    FString PromptFstring(Prompt.c_str());
  //  UE_LOG(LlamaLog, Log, TEXT("prompt originale %s "), *PromptFstring);
    const auto StartTime = ggml_time_us();

    bGenerationActive = true;

    if (!Prompt.empty())
    {
        int32 TokensProcessed = ProcessPrompt(Prompt);
    }

    std::string Response;

    const llama_vocab* Vocab = llama_model_get_vocab(LlamaModel);

    llama_batch Batch;

    llama_token NewTokenId;
    int32 NDecoded = 0;

    // check if we have enough space in the context to evaluate this batch - might need to be inside loop
    int NContext = llama_n_ctx(Context);
    int NContextUsed = llama_kv_self_used_cells(Context);
    bool bEOGExit = false;

    while (bGenerationActive) //processing can be aborted by flipping the boolean
    {
        //Common sampler is a bit faster
        if (CommonSampler)
        {
            NewTokenId = common_sampler_sample(CommonSampler, Context, -1); //sample using common sampler
            common_sampler_accept(CommonSampler, NewTokenId, true);
        }
        else
        {
            if(PoolingTypeOfModel==-1) NewTokenId = llama_sampler_sample(Sampler, Context, -1);
       
        }

        // is it an end of generation?
        if (llama_vocab_is_eog(Vocab, NewTokenId))
        {
            bEOGExit = true;
            break;
        }
        std::string Piece;
        if (Vocab) 
        {
          // UE_LOG(LlamaLog, Log, TEXT("Vocab IS  Valid"));
            if (NewTokenId)
            {
             //   UE_LOG(LlamaLog, Log, TEXT("NewTokenID IS  Valid"));
                // convert the token to a string, print it and add it to the response
                Piece = common_token_to_piece(Vocab, NewTokenId, true);
            }
            else
            {
             //   UE_LOG(LlamaLog, Log, TEXT("NewTokenID IS NOT Valid"));
                return "tokendid error";
                
            }
    
        } 
        else
        {
         //   UE_LOG(LlamaLog, Log, TEXT("Vocab IS NOT Valid"));
            return "vocab error";
        }
        Response += Piece;
      
        NDecoded += 1;
        if (NDecoded % 10 == 0) {

        }

        if (NContextUsed + NDecoded > NContext)
        {
           
            FString ErrorMessage = FString::Printf(TEXT("Context size %d exceeded on generation. Try increasing the context size and re-run prompt"), NContext);
            if (ErrorFound)
            {
                ErrorFound(*ErrorMessage);
            }
            ResponseStatus = ELLMResponseStatus::ContextOversize;
            EmitErrorMessage(ErrorMessage, 31, __func__);
            FString stringa(Response.c_str());
            UE_LOG(LlamaLog, Log, TEXT("Response %s "), *stringa);
            return Response;
        }

        if (OnTokenGenerated)
        {
            OnTokenGenerated(Piece);
        }

        // prepare the next batch with the sampled token
        Batch = llama_batch_get_one(&NewTokenId, 1);

        if (llama_decode(Context, Batch))
        {
            bGenerationActive = false;
      
            FString ErrorMessage = TEXT("Failed to decode. Could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
            if (ErrorFound)
            {
				ErrorFound(*ErrorMessage);
            }
            ResponseStatus = ELLMResponseStatus::FailedToDecode;
            EmitErrorMessage(ErrorMessage, 32, __func__);
            //Return partial response
            FString stringa(Response.c_str());
            UE_LOG(LlamaLog, Log, TEXT("Response %s "), *stringa);
            return Response;
        }

        //sleep pacing
        if (LastLoadedParams.Advanced.TokenGenerationPacingSleep > 0.f)
        {
            FPlatformProcess::Sleep(LastLoadedParams.Advanced.TokenGenerationPacingSleep);
        }
    }

    bGenerationActive = false;

    const auto StopTime = ggml_time_us();
    const float Duration = (StopTime - StartTime) / 1000000.0f;

    if (bAppendToMessageHistory)
    {
        //Add the response to our templated messages
        Messages.push_back({ RoleForEnum(EChatTemplateRole::Assistant), _strdup(Response.c_str()) });

        //Sync ContextHistory
        FilledContextCharLength = ApplyTemplateToContextHistory(false);
    }

    if (OnGenerationComplete)
    {
        OnGenerationComplete(Response, Duration, NDecoded, NDecoded / Duration);
    }
    FString stringa(Response.c_str());
    UE_LOG(LlamaLog, Log, TEXT("Response %s "), *stringa);
	ResponseStatus = ELLMResponseStatus::Success;
    return Response;
}

void FLlamaInternal::EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode, const FString& FunctionName)
{
    UE_LOG(LlamaLog, Error, TEXT("[%s error %d]: %s"), *FunctionName, ErrorCode, *ErrorMessage);
    FString ErrorMessageCopy = ErrorMessage;
    int32 ErrorCodeCopy = ErrorCode;
	//this for eviting some crush whene have a error 
    if (OnError)
    {
        AsyncTask(ENamedThreads::BackgroundThreadPriority, [this, ErrorMessageCopy, ErrorCodeCopy]()
            {
                // WE ARE ON GAMETHREAD            
                if (OnError)
                {
                    OnError(ErrorMessageCopy, ErrorCodeCopy);
                }
            });
    }
    
}

//NB: this function will apply out of range errors in log, this is normal behavior due to how templates are applied
int32 FLlamaInternal::ApplyTemplateToContextHistory(bool bAddAssistantBOS)
{
    return ApplyTemplateFromMessagesToBuExample3er(Template, Messages, ContextHistory, bAddAssistantBOS);
}

int32 FLlamaInternal::ApplyTemplateFromMessagesToBuExample3er(const std::string& InTemplate, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuExample3er, bool bAddAssistantBoS)
{
    int32 NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
        bAddAssistantBoS, ToBuExample3er.data(), ToBuExample3er.size());

    //Resize if ToBuExample3er can't hold it
    if (NewLen > ToBuExample3er.size())
    {
        ToBuExample3er.resize(NewLen);
        NewLen = llama_chat_apply_template(InTemplate.c_str(), FromMessages.data(), FromMessages.size(),
            bAddAssistantBoS, ToBuExample3er.data(), ToBuExample3er.size());
    }
    if (NewLen < 0)
    {
        EmitErrorMessage(TEXT("Failed to apply the chat template ApplyTemplateFromMessagesToBuExample3er."), 101, __func__);
    }
    return NewLen;
}

const char* FLlamaInternal::RoleForEnum(EChatTemplateRole Role)
{
    if (Role == EChatTemplateRole::User)
    {
        return "user";
    }
    else if (Role == EChatTemplateRole::Assistant)
    {
        return "assistant";
    }
    else if (Role == EChatTemplateRole::System)
    {
        return "system";
    }
    else {
        return "unknown";
    }
}

//from https://github.com/ggml-org/llama.cpp/blob/master/examples/embedding/embedding.cpp
void FLlamaInternal::BatchDecodeEmbedding(llama_context* InContext, llama_batch& Batch, float* Output, int NSeq, int NEmbd, int EmbdNorm)
{
    const enum llama_pooling_type pooling_type = llama_pooling_type(InContext);
    const struct llama_model* model = llama_get_model(InContext);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_self_clear(InContext);

    // run model

    //Debug info
 //   UE_LOG(LlamaLog, Log, TEXT("%hs: n_tokens = %d, n_seq = %d"), __func__, Batch.n_tokens, NSeq);
   // UE_LOG(LlamaLog, Log, TEXT("inizio encoder e decoder"));
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model))
    {
        // encoder-only model
        if (llama_encode(InContext, Batch) < 0)
        {
            UE_LOG(LlamaLog, Error, TEXT("%hs : failed to encode"), __func__);
        }
    }
    else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model))
    {
        // decoder-only model
        if (llama_decode(InContext, Batch) < 0)
        {
            UE_LOG(LlamaLog, Log, TEXT("%hs : failed to decode"), __func__);
        }
    }
    //UE_LOG(LlamaLog, Log, TEXT("inizio for nbacth "));
    for (int i = 0; i < Batch.n_tokens; i++)
    {
        if (Batch.logits && !Batch.logits[i])
        {
            continue;
        }

        const float* Embd = nullptr;
        int EmbdPos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE)
        {
            // try to get token embeddings
            Embd = llama_get_embeddings_ith(InContext, i);
            EmbdPos = i;
            GGML_ASSERT(Embd != NULL && "failed to get token embeddings");
            if (Embd == NULL)UE_LOG(LlamaLog, Log, TEXT("failed to get token embeddings "));
        }
        else if (Batch.seq_id)
        {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            Embd = llama_get_embeddings_seq(InContext, Batch.seq_id[i][0]);
            EmbdPos = Batch.seq_id[i][0];
            GGML_ASSERT(Embd != NULL && "failed to get sequence embeddings");
            if(Embd==NULL) UE_LOG(LlamaLog, Log, TEXT("failed to get sequence embeddings"));
        }
        else
        {
            //NB: this generally won't work, we should crash here.
            Embd = llama_get_embeddings(InContext);
            UE_LOG(LlamaLog, Log, TEXT("llama_get_embeddings"));
        }

        float* Out = Output + EmbdPos * NEmbd;
        common_embd_normalize(Embd, Out, NEmbd, EmbdNorm);
       // UE_LOG(LlamaLog, Log, TEXT(" common_embd_normalize"));
    }
}

void FLlamaInternal::BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id)
{
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++)
    {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

FLlamaInternal::FLlamaInternal()
{

}

FLlamaInternal::~FLlamaInternal()
{
    OnTokenGenerated = nullptr;
    UnloadModel();
    llama_backend_free();
}

void FLlamaInternal::SetPoolingMode(int NewPoolingMode, int NewPoolingType)
{
    PoolingMode = NewPoolingMode;
    PoolingTypeOfModel = NewPoolingType;
}
/*
static void batch_add_seq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}*/
static void batch_process(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    //llama_memory_clear(llama_get_memory(ctx), false);
    // check batch tokens
    int64 bt = batch.n_tokens;
    int64 nctx = llama_n_ctx(ctx);
    if (bt > nctx)
    {
        UE_LOG(LogTemp, Error, TEXT("Errore critico: la dimensione del batch (%d) supera la dimensione del contesto (%d)!"), batch.n_tokens, llama_n_ctx(ctx));
      
    }
    // run model
   // UE_LOG(LlamaLog, Log, TEXT("%s: n_tokens = %d, n_seq = %d"), __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        UE_LOG(LlamaLog, Log, TEXT(" failed to process llama_decode on batch_process "));
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float* embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                UE_LOG(LlamaLog, Log, TEXT("failed to get embeddings for token %d"),i);
                return ;
            }
        }

        float* out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, 2);
    }
}

float  FLlamaInternal::CalculateCosSim(std::string query, std::string DocChunk, uint32_t n_batch,int32 NEmbd)
{
    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
   // std::string query = "query: ";// la domanda che porremo 
   // query += 

    std::vector<int32_t> query_tokens = common_tokenize(Context, query, true);// domanda convertita in token
    BatchAddSeq(query_batch, query_tokens, 0);
    std::vector<float> query_emb(NEmbd, 0);// query embeddizata
    batch_process(Context, query_batch, query_emb.data(), 1, NEmbd);
    struct llama_batch DocChunk_batch = llama_batch_init(n_batch, 0, 1);
    std::vector<int32_t> DocChunk_tokens = common_tokenize(Context, DocChunk, true);// domanda convertita in token
    BatchAddSeq(DocChunk_batch, DocChunk_tokens, 0);
    std::vector<float> DocChunk_emb(NEmbd, 0);// query embeddizata
    batch_process(Context, DocChunk_batch, DocChunk_emb.data(), 1, NEmbd);
    float sim = common_embd_similarity_cos(DocChunk_emb.data(), query_emb.data(), NEmbd);
    float result = sim;// risultato da restituire. 
    llama_batch_free(query_batch);
    llama_batch_free(DocChunk_batch);
    DocChunk_emb.~vector();
    query_emb.~vector();
    llama_backend_free();
    return result;
}

FString FLlamaInternal::RetriveFromEmbedding(const FString& Text)
{   // path dei documenti
   
    FString Path = FPaths::ProjectContentDir()+ "/RAG/Testo.txt";
    FString PathExample1 = FPaths::ProjectContentDir() + "/RAG/TestoExample1.txt";
    FString PathExample2 = FPaths::ProjectContentDir() + "/RAG/TestoExample2.txt";
    FString PathExample3 = FPaths::ProjectContentDir() + "/RAG/TestoExample3.txt";
    //check 
    if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*Path))      
    {
        UE_LOG(LogTemp, Error, TEXT("Il percorso %s non è valido"),*Path);
        return FString();
        }

  //  FString TextStr="Il Colosseo, originariamente conosciuto come Anfiteatro Flavio, è il più grande anfiteatro del mondo e un'icona immortale di Roma. La sua costruzione, iniziata dall'imperatore Vespasiano nel 72 d.C. e completata da suo figlio Tito nell'80 d.C., rappresenta un capolavoro dell'ingegneria romana. Con la sua complessa struttura di archi e volte in calcestruzzo e travertino, era in grado di ospitare oltre 50.000 spettatori. Al suo interno si svolgevano spettacoli grandiosi, dai combattimenti tra gladiatori alle cacce ad animali esotici (venationes), fino a naumachie, vere e proprie battaglie navali simulate allagando l'arena. Il suo avanzato sistema di corridoi e uscite, il vomitorium, permetteva di evacuare la folla in pochi minuti, un concetto ancora oggi alla base della progettazione degli stadi moderni./n La Grande Piramide di Giza, nota anche come Piramide di Cheope, è l'unica delle Sette Meraviglie del Mondo Antico ad essere giunta quasi intatta fino ai giorni nostri. Costruita come monumento funebre per il faraone Cheope intorno al 2560 a.C., questa colossale struttura ha detenuto il record di edificio più alto del mondo per oltre 3.800 anni. È composta da circa 2,3 milioni di blocchi di pietra, ciascuno del peso medio di 2,5 tonnellate, assemblati con una precisione millimetrica che sconcerta ancora oggi gli ingegneri. Le teorie sulla sua costruzione sono numerose e dibattute, ma tutte concordano sull'incredibile livello di organizzazione logistica e di conoscenza astronomica e matematica posseduto dagli antichi Egizi, evidente nell'allineamento quasi perfetto della piramide con i punti cardinali. /n La Grande Muraglia Cinese è una delle più straordinarie opere di ingegneria militare mai realizzate.Non si tratta di un unico muro continuo, ma di una serie di fortificazioni, mura, torri di avvistamento e fortezze costruite e ricostruite nel corso di secoli, a partire dal VII secolo a.C.La sezione più famosa fu edificata durante la dinastia Ming(1368 - 1644) per proteggere l'impero cinese dalle incursioni delle tribù nomadi della steppa. Estendendosi per oltre 21.000 chilometri, la sua costruzione ha richiesto l'impiego di milioni di soldati e operai, utilizzando materiali disponibili localmente come terra battuta, legno, mattoni e pietra.Più che una barriera invalicabile, funzionava come un sistema di difesa integrato, permettendo un rapido spostamento di truppe e l'invio di segnali di fumo per allertare di un imminente attacco. /n Machu Picchu, arroccata sulle Ande peruviane a quasi 2.500 metri di altitudine, è la più celebre e misteriosa città del grande Impero Inca.Costruita intorno al 1450 sotto il regno dell'imperatore Pachacútec, si pensa fosse una tenuta reale o un santuario religioso. Abbandonata circa un secolo dopo, durante la conquista spagnola, rimase nascosta al mondo esterno fino alla sua riscoperta nel 1911. La sua architettura è un prodigio di integrazione con il paesaggio: gli edifici in pietra sono realizzati con la tecnica ashlar, che prevede l'incastro di blocchi di granito lavorati con una precisione tale da non richiedere l'uso di malta. Il complesso sistema di terrazzamenti agricoli, canali d'acqua e osservatori astronomici dimostra una profonda conoscenza dell'ingegneria idraulica e dell'astronomia.";
    
    // la stringa da cui faremo poi il parser
    FString TextStr = "";
    FString TextExample2 = "";
    FString TextExample1 = "";
    FString TextExample3 = "";

    
   
    TextExample1 = "";
    TextExample2 = ""; 
    TextExample3 = ""
 
    llama_backend_init();
    if (Context) {
        UE_LOG(LogTemp, Warning, TEXT("Context Exist  "));
    const uint32_t n_batch = llama_n_batch(Context);  // max batch size
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);//  inizializiamo il contentitorie di dimensioni n batch 
     // allocate output
    int32 NEmbd = llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.
    // calcolo a quale document ofar riferimento. 
    std::string query = std::string(TCHAR_TO_UTF8(*Text));

    TArray<Chunk>FirstLevelRetrive;

   // UE_LOG(LogTemp, Log, TEXT("Similarity text Example1 is %f"), Sim);
    Chunk CExample1;
    CExample1.Text = TextExample1; //"Example1";
  //  CExample1.Similitarity = Sim;
    CExample1.Path = PathExample1;

    Chunk CExample2;
    CExample2.Text = TextExample2;// "Example2";
   // CExample2.Similitarity = Sim;
    CExample2.Path = PathExample2;

  //  std::string TextExample3_c = std::string(TCHAR_TO_UTF8(*TextExample3))
    Chunk CExample3;
    CExample3.Text = TextExample3;// "Example3";
    CExample3.Path = PathExample3;
  
    FirstLevelRetrive.Add(CExample1);
    FirstLevelRetrive.Add(CExample2);
    FirstLevelRetrive.Add(CExample3);
    //embedd di tutti e tre i riassunti 
    FString ManualSelected;
    for (auto& ChunckT : FirstLevelRetrive) {
        llama_kv_cache_clear(Context);
        std::string Chunk_text_string = std::string(TCHAR_TO_UTF8(*ChunckT.Text));
        struct llama_batch Chunk_batch = llama_batch_init(n_batch, 0, 1);
        std::vector<int32_t> Chunk_text_string_tokens = common_tokenize(Context, Chunk_text_string, true);// domanda convertita in token
        BatchAddSeq(Chunk_batch, Chunk_text_string_tokens, 0);
     
        std::vector<float> Chunk_emb(NEmbd, 0);// query embeddizata
        batch_process(Context, Chunk_batch, Chunk_emb.data(), 1,NEmbd);
        for (int j = 0; j < Chunk_emb.size(); j++)
        {
            ChunckT.Embedding.Add(Chunk_emb[j]);
        }
        llama_batch_free(Chunk_batch);
    
    }
    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
    //query = "query: ";// la domanda che porremo 
    query += std::string(TCHAR_TO_UTF8(*Text));
    std::vector<int32_t> query_tokens = common_tokenize(Context, query, true);// domanda convertita in token

    llama_kv_cache_clear(Context);
    BatchAddSeq(query_batch, query_tokens, 0);
    std::vector<float> query_emb(NEmbd, 0);// query embeddizata
    UE_LOG(LogTemp, Log, TEXT("batch process for query "));
    batch_process(Context, query_batch, query_emb.data(), 1, NEmbd);

    for (auto& ChunckT : FirstLevelRetrive) {
      
        float sim = common_embd_similarity_cos(ChunckT.Embedding.GetData(), query_emb.data(), NEmbd);
        ChunckT.Similitarity = sim;
      //  UE_LOG(LogTemp, Log, TEXT("Risultati parziali %f: %s"), sim, *ChunckT.Text);
    }



    FirstLevelRetrive.Sort([](const Chunk& A, const Chunk& B) {return A.Similitarity > B.Similitarity; });
    UE_LOG(LogTemp, Log, TEXT("Risultati FirstLevel %f: %s"), FirstLevelRetrive[0].Similitarity, *FirstLevelRetrive[0].Text);
    if (!Example3ileHelper::LoadFileToString(TextStr, *FirstLevelRetrive[0].Path))
    {
        UE_LOG(LogTemp, Error, TEXT("Il percorso %s non è valido"), *FirstLevelRetrive[0].Path);
        return FString();
    }
    TArray<FString> Parsed;
    TextStr.ParseIntoArray(Parsed, TEXT("@"));
    n_chunks = Parsed.Num();
    UE_LOG(LogTemp, Log, TEXT(" number of chunks is %d"), n_chunks);
    const llama_vocab* vocab = llama_model_get_vocab(LlamaModel);// vocavolario del modello 
    std::vector<float> embeddings(n_chunks * NEmbd, 0); // embeddings numero
    float * emb = embeddings.data(); // puntatore embeddings
    // break into batches
    llama_kv_cache_clear(Context);
    for (auto& Pars : Parsed) {
        //la versione string di FString
        std::string textdata= std::string(TCHAR_TO_UTF8(*Pars));
        //token del testo con contesto.
        auto inp = common_tokenize(Context, textdata, true, false);
        if (inp.size() > n_batch) {
            UE_LOG(LogTemp, Error, TEXT(" chunk size exceeds batch size increase batch size and re-run  "));
            return FString();
        }
        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab))) {
            inp.push_back(llama_vocab_eos(vocab));
        }
        // Chunk creato e immesso nell'array
        Chunk chunk;
        chunk.Text = Pars;
        chunk.tokens = inp;
        IndexedChunks.Add(chunk);
    }
    UE_LOG(LogTemp, Log, TEXT(" number of Indexed Chunks is %d, and NEmbd %d"), IndexedChunks.Num(),NEmbd);
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) 
    {
        auto& inp = IndexedChunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float* out = emb + p * NEmbd;
            batch_process(Context, batch, out, s, NEmbd);
            common_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        BatchAddSeq(batch, inp, s);
        s += 1;
    }
    // final batch
    float* out = emb + p * NEmbd;
    batch_process(Context, batch, out, s, NEmbd);
    UE_LOG(LogTemp, Log, TEXT("batch process eseguito "));
    for (int i = 0; i < n_chunks; i++) {
        std::vector<float> vettore=  std::vector<float>(emb + i * NEmbd, emb + (i + 1) * NEmbd);
        for (int j = 0; j < vettore.size(); j++) 
        {
            IndexedChunks[i].Embedding.Add(vettore[j]);
        }
        // clear tokens as they are no longer needed
        IndexedChunks[i].tokens.clear();
    }
         std::vector<std::pair<int, float>> similarities;// la similarità , con indice e valoro della simirialità coseinduale
        for (int i = 0; i < n_chunks; i++) {

            float sim = common_embd_similarity_cos(IndexedChunks[i].Embedding.GetData(), query_emb.data(), NEmbd);
            similarities.push_back(std::make_pair(i, sim));
            IndexedChunks[i].Similitarity = sim;
            IndexedChunks[i].SimilitarityIndex = i;
        }

    
      //  LOG("Top %d similar chunks:\n", params.sampling.top_k);
        for (int i = 0; i < similarities.size(); i++) {
            //     LOG("filename: %s\n", IndexedChunks[similarities[i].first].filename.c_str());
           //      LOG("filepos: %lld\n", (long long int) chunks[similarities[i].first].filepos);

            FString Stringauscita = IndexedChunks[i].Text;
         //   UE_LOG(LogTemp, Log, TEXT("similarity: %f"), similarities[i].second);
           // UE_LOG(LogTemp, Log, TEXT("textdata: %s"), *Stringauscita);

        }
        FString ResultText= FirstLevelRetrive[0].Text;// il Text che porteremo alla fine.
        IndexedChunks.Sort([](const Chunk& A, const Chunk& B) {return A.Similitarity > B.Similitarity; });
        NumberOfChunksToGet = 2;
        for (int i = 0; i < 2; i++) {// stampo i primi 5 
           // UE_LOG(LogTemp, Log, TEXT("Risultati %f: %s"), IndexedChunks[i].Similitarity, *IndexedChunks[i].Text);
            ResultText += IndexedChunks[i].Text;
        }
        llama_batch_free(query_batch);
        llama_batch_free(batch);
        llama_backend_free();
        UE_LOG(LogTemp, Warning, TEXT("Clearing KV Cache before text generation..."));
        llama_kv_cache_clear(Context);
        //Full Reset
        ContextHistory.clear();
        Messages.clear();

        llama_kv_self_clear(Context);
        FilledContextCharLength = 0;
        IndexedChunks.Empty();

        // FMemory::Free(&TextStr);
        return ResultText;
    }
    }else         UE_LOG(LogTemp, Error, TEXT("Context don't Exist  "));
 
    return "NULL";
    
}

void FLlamaInternal::FromJsonToEmbeddingsJson(const FString& Input, const FString& Output)
{
    //CreateDatabase();
  
    FString InputFileName = Input; //"database_vettoriale_ue5.json";
    FString FileName = "";
    FString OutputFileName = Output; //"database_vettoriale_ue5_emb.json";
    // 1. Ottieni il percorso completo del file
    // Si aspetta che il file sia nella cartella Content del progetto
    const FString JsonFilePath = FPaths::ProjectContentDir() + InputFileName;

    // 2. Leggi il contenuto del file in una stringa
    FString JsonString;
    if (!Example3ileHelper::LoadFileToString(JsonString, *JsonFilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("Impossibile caricare il file JSON: %s"), *JsonFilePath);
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("File JSON caricato con successo. Inizio parsing..."));
    TArray<Example3romJson> OutChunkDatabase;
    // 3. Fai il parsing della stringa JSON e convertila direttamente in un TArray della nostra USTRUCT
    // Questa funzione è potentissima: fa tutto il lavoro per noi!

    if (FJsonObjectConverter::JsonArrayStringToUStruct(JsonString, &OutChunkDatabase, 0, 0))
    {
        UE_LOG(LogTemp, Log, TEXT("Parsing del JSON riuscito! Caricati %d chunk nel database."), OutChunkDatabase.Num());

    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Errore durante il parsing della stringa JSON."));
        return;
    }
    n_chunks = OutChunkDatabase.Num();
    if (OutChunkDatabase.Num() > 0) {

        UE_LOG(LogTemp, Log, TEXT(" number of outchunkdatabase is %d"), OutChunkDatabase.Num());
        llama_backend_init();
       // FVectorDBParams DBParams;
       // DBParams.Dimensions = llama_model_n_embd(LlamaModel); // Assumiamo che il plugin abbia questa funzione
        //DBParams.MaxElements = 4096;// OutChunkDatabase.Num();
      //  VectorDB->InitializeDB();
        for (int i = 0; i < OutChunkDatabase.Num(); i++)
        {
            TArray<float> EmbeddingMatrix;
            llama_kv_cache_clear(Context);
            //UE_LOG(LogTemp, Log, TEXT("elemento %d testo  %s"), OutChunkDatabase[i].id, *OutChunkDatabase[i].text);
            std::string textdata = std::string(TCHAR_TO_UTF8(*OutChunkDatabase[i].text));
            std::vector<float> Embeddings;
            GetPromptEmbeddings(textdata, Embeddings);
            UE_LOG(LogTemp, Log, TEXT("elemento %d size embedding  %d"), OutChunkDatabase[i].id, Embeddings.size());
            for (int j = 0; j < Embeddings.size(); j++) 
            {
                EmbeddingMatrix.Add(Embeddings[j]);
            }
            // Copia i dati in modo più diretto ed eExample3iciente
            OutChunkDatabase[i].Embedding = TArray<float>(Embeddings.data(), Embeddings.size());
         //   VectorDB->AddVectorEmbeddingStringPair(OutChunkDatabase[i].Embedding, OutChunkDatabase[i].text);
            EmbeddingMatrix.Empty();
        }
        //if (Context) 
        if(false)
        {
            UE_LOG(LogTemp, Warning, TEXT("Context Exist  "));
            const uint32_t n_batch = llama_n_batch(Context);  // max batch size
            struct llama_batch batch = llama_batch_init(n_batch, 0, 1);//  inizializiamo il contentitorie di dimensioni n batch 
            // allocate output
            int32 NEmbd = llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.
            // calcolo a quale document ofar riferimento. 
            const llama_vocab* vocab = llama_model_get_vocab(LlamaModel);// vocavolario del modello 
            std::vector<float> embeddings(n_chunks * NEmbd, 0); // embeddings numero
            float* emb = embeddings.data(); // puntatore embeddings
           // llama_kv_cache_clear(Context);
            int processed_chunks_count = 0; // Contatore per l'output buExample3er
            int chunks_in_current_batch = 0; // Contatore per il batch corrente
            for (int i = 0; i < OutChunkDatabase.Num(); i++)
            {
                llama_kv_cache_clear(Context);
                UE_LOG(LogTemp, Log, TEXT("elemento %d testo  %s"), OutChunkDatabase[i].id, *OutChunkDatabase[i].text);

                std::string textdata = std::string(TCHAR_TO_UTF8(*OutChunkDatabase[i].text));
                //struct llama_batch Chunk_batch = llama_batch_init(n_batch, 0, 1);
                //token del testo con contesto.
                auto inp = common_tokenize(Context, textdata, true, false);
                if (inp.size() > n_batch) {
                    UE_LOG(LogTemp, Error, TEXT(" chunk size exceeds batch size increase batch size and re-run  "));
                    continue;
                }
                if (batch.n_tokens + inp.size() > (uint32)n_batch) {
                    UE_LOG(LogTemp, Log, TEXT("Batch pieno. Processo %d chunk."), chunks_in_current_batch);
                    float* out_ptr = emb + processed_chunks_count * NEmbd;
                    batch_process(Context, batch, out_ptr, chunks_in_current_batch, NEmbd);

                    // Aggiorna i contatori e pulisci il batch per il prossimo giro
                    processed_chunks_count += chunks_in_current_batch;
                    chunks_in_current_batch = 0;
                    common_batch_clear(batch);
              
                }
                BatchAddSeq(batch, inp, chunks_in_current_batch);
                chunks_in_current_batch++;
                //if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab))) {
                //    inp.push_back(llama_vocab_eos(vocab));
              //  }

                // add eos if not present


               // UE_LOG(LogTemp, Log, TEXT(" number of outchunkdatabase is %d, and NEmbd %d turn %d"), OutChunkDatabase.Num(), NEmbd, i);

                UE_LOG(LogTemp, Log, TEXT(" turn %d tokens %d "), i, inp.size());
                //  int p = 0; // number of prompts processed already
               //   int s = 0; // number of prompts in current batch
             /*    // for (int k = 0; k < n_chunks; k++)
                  {
                    //  auto& inp = IndexedChunks[k].tokens;
                      const uint64_t n_toks = inp.size();

                      // encode if at capacity
                 //     if (batch.n_tokens + n_toks > n_batch) {
                        // UE_LOG(LogTemp, Log, TEXT("batch_process in progress in cycle"));
                          float* out = emb + p * NEmbd;
                          batch_process(Context, batch, out, s, NEmbd);
                          common_batch_clear(batch);
                          p += s;
                          s = 0;
                      }


                      // add to batch
                      BatchAddSeq(batch, inp, s);
                      s += 1;
                  }
                  */
                  // final batch

                float* out = emb + processed_chunks_count * NEmbd;
                //  UE_LOG(LogTemp, Log, TEXT("batch process in progress after the cycle "));
                batch_process(Context, batch, out, chunks_in_current_batch, NEmbd);
                inp.clear();
            }

                //UE_LOG(LogTemp, Log, TEXT("batch process eseguito "));
                for (int t = 0; t < n_chunks; t++) {
                    std::vector<float> vettore = std::vector<float>(emb + t * NEmbd, emb + (t + 1) * NEmbd);
                 //   UE_LOG(LogTemp, Log, TEXT("vettore size %d "),vettore.size());
                    for (int j = 0; j < vettore.size(); j++)
                    {
                        OutChunkDatabase[t].Embedding.Add(vettore[j]);

                    }
                    // clear tokens as they are no longer needed
                   //
           
                }
               // UE_LOG(LogTemp, Log, TEXT("trasferimento da tokens ad embedding  "));
            

        llama_batch_free(batch);
        }
       // else { UE_LOG(LogTemp, Warning, TEXT("Context don't  Exist  ")); }
        SaveDataToJsonFile_Incremental(OutChunkDatabase, OutputFileName);
       // VectorDB->SaveIndex(TEXT("testbin.bin"),"");
    }
}

bool FLlamaInternal::SaveDataToJsonFile( TArray<Example3romJson>& DataToSave,  FString& FileName)
{
    // --- PASSO 1: CONVERTIRE L'ARRAY DI STRUCT IN UNA STRINGA JSON ---

    FString JsonString="[";
    // Questa è la funzione magica che fa tutto il lavoro per noi!
    // Prende il nostro array di struct e lo trasforma in una stringa JSON ben formattata.
    for (int i = 0; i < DataToSave.Num(); i++) {
        if (i != 0)JsonString += ",";
        FString NewString;
        if (!FJsonObjectConverter::UStructToJsonObjectString(DataToSave[i], NewString))
        {//UStructArrayToJsonString
            UE_LOG(LogTemp, Error, TEXT("Impossibile convertire l'array di struct in una stringa JSON."));
            return false;
        }
        UE_LOG(LogTemp, Error, TEXT("JsonString %s"), *JsonString);
        JsonString += NewString;
    }
    JsonString += "]";
//    UE_LOG(LogTemp, Log, TEXT("Conversione in JSON riuscita. Stringa generata:\n%s"), *JsonString);


    // --- PASSO 2: SALVARE LA STRINGA JSON SU FILE ---

    // Decidiamo dove salvare il file. La cartella "Saved" è un buon posto per i dati generati.
    const FString OutputFilePath = FPaths::ProjectSavedDir() + TEXT("/") + FileName;

    // Example3ileHelper scrive la nostra stringa nel file specificato
    if (Example3ileHelper::SaveStringToFile(JsonString, *OutputFilePath))
    {
        UE_LOG(LogTemp, Log, TEXT("File JSON salvato con successo in: %s"), *OutputFilePath);
        return true;
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Impossibile salvare il file JSON in: %s"), *OutputFilePath);
        return false;
    }
}

FString FLlamaInternal::RetriveFromJson(const FString& Text,const FString& Json,int NChuncksOut)
{
     FString ResultText;
    if (!Context) {
        return "Error on Context";
    }
    FString InputFileName = Json;// "database_vettoriale_Embedded.json";
    const FString JsonFilePath = FPaths::ProjectSavedDir() + InputFileName;

    // 2. Leggi il contenuto del file in una stringa

      UE_LOG(LogTemp, Log, TEXT("File JSON caricato con successo. Inizio parsing..."));
    TArray<Example3romJson> OutChunkDatabase;
    //AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, &OutChunkDatabase, JsonFilePath, JsonString]()
  //  {
        // 3. Fai il parsing della stringa JSON e convertila direttamente in un TArray della nostra USTRUCT
    // Questa funzione è potentissima: fa tutto il lavoro per noi!
    //    FString JsonString;
	bool errorparsing = false;
    AsyncTask(ENamedThreads::AnyBackgroundHiPriTask, [this, &JsonFilePath, &OutChunkDatabase,&errorparsing]() mutable {
            FString JsonString;
            if (!Example3ileHelper::LoadFileToString(JsonString, *JsonFilePath))
            {
                UE_LOG(LogTemp, Error, TEXT("Impossibile caricare il file JSON: %s"), *JsonFilePath);
             //   return "Error on Json";
            }
            if (FJsonObjectConverter::JsonArrayStringToUStruct(JsonString, &OutChunkDatabase, 0, 0))
            {
                UE_LOG(LogTemp, Log, TEXT("Parsing del JSON riuscito! Caricati %d chunk nel database."), OutChunkDatabase.Num());

            }
            else
            {
                UE_LOG(LogTemp, Error, TEXT("Errore durante il parsing della stringa JSON."));
                errorparsing=true;
            //  return "Error on Parsing Json";
                
            }
	});
	if (errorparsing) {
		UE_LOG(LogTemp, Error, TEXT("Errore durante il parsing della stringa JSON."));
		return "Error on Parsing Json";
	}
    std::string query = std::string(TCHAR_TO_UTF8(*Text));
    const uint32_t n_batch = llama_n_batch(Context);  // max batch size
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);//  inizializiamo il contentitorie di dimensioni n batch
    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
    int32 NEmbd = llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.
    //query = "query: ";// la domanda che porremo
    //query += std::string(TCHAR_TO_UTF8(*Text));
    std::vector<int32_t> query_tokens = common_tokenize(Context, query, true);// domanda convertita in token
    //batch_add_seq(query_batch, query_tokens, 0);
    llama_kv_cache_clear(Context);
    //BatchAddSeq(query_batch, query_tokens, 0); commentato per aggiugnere get PromptEmbeddigns
    std::vector<float> query_emb;//(NEmbd, 0);// query embeddizata cambiata per mettere getpromptembeddings
    
    //UE_LOG


        GetPromptEmbeddings(query, query_emb);

  /*  for (auto& ChunckT : OutChunkDatabase) {

        float sim = common_embd_similarity_cos(ChunckT.Embedding.GetData(), query_emb.data(), NEmbd);
        ChunckT.Similitarity = sim;
        UE_LOG(LogTemp, Log, TEXT("Risultati parziali %f: %s"), sim, *ChunckT.text);
    }*/
    FCriticalSection CritSec; // Lucchetto per proteggere l'array
	TArray<Example3romJson> ChunksCopy; // Copia dell'array per l'elaborazione parallela
    ChunksCopy = OutChunkDatabase;
    std::vector<float> QueryEmbeddingCopy;
	QueryEmbeddingCopy = query_emb;// Copia dell'embedding della query per l'elaborazione parallela
	int32 NEmbdCopy = NEmbd; // Grandezza dell'embedding per l'elaborazione parallela
 
     for (auto& ChunckT : OutChunkDatabase) {

            float sim = common_embd_similarity_cos(ChunckT.Embedding.GetData(), query_emb.data(), NEmbd);
            ChunckT.Similitarity = sim;
            UE_LOG(LogTemp, Log, TEXT("Risultati parziali %f: %s"), sim, *ChunckT.text);
        }
    
    OutChunkDatabase.Sort([](const Example3romJson& A, const Example3romJson& B) {return A.Similitarity > B.Similitarity; });
    // UE_LOG(LogTemp, Log, TEXT("Risultati  sim : %f text : %s"), OutChunkDatabase[0].Similitarity, *OutChunkDatabase[0].text);
   
    for (int i = 0; i < NChuncksOut; i++)
    {
        ResultText += OutChunkDatabase[i].text;
        //	UE_LOG(LogTemp, Log, TEXT("Risultati nchucnk %f: %s"), OutChunkDatabase[i].Similitarity, *OutChunkDatabase[i].text);
    }
    return ResultText;
}

void FLlamaInternal::CreateDatabase() {
    
    VectorDatabase->InitializeDB();
    //Database.BasicsTest();

}


bool FLlamaInternal::CheckContext()
{   
    if (Context) return true;
    else return false;
}


bool FLlamaInternal::SaveDataToJsonFile_Incremental(TArray<Example3romJson>& DataToSave, FString& FileName)
{
    // --- PASSO 1: PREPARARE IL FILE DI DESTINAZIONE ---

    // Definiamo il percorso completo del file nella cartella "Saved" del progetto.
    const FString OutputFilePath = FPaths::ProjectSavedDir() + TEXT("/") + FileName;

    // Se il file esiste già da una sessione precedente, lo eliminiamo.
    // Questo è FONDAMENTALE per assicuraci di iniziare sempre con un file JSON valido e pulito.
    IFileManager::Get().Delete(*OutputFilePath, false, true, true);

    // Gestiamo il caso in cui l'array di dati sia vuoto.
    if (DataToSave.Num() == 0)
    {
        // Se non ci sono dati, scriviamo semplicemente un array JSON vuoto "[]" e terminiamo.
        UE_LOG(LogTemp, Warning, TEXT("L'array di dati è vuoto. Salvataggio di un file JSON vuoto."));
        return Example3ileHelper::SaveStringToFile(TEXT("[]"), *OutputFilePath);
    }

    // --- PASSO 2: SCRIVERE I DATI SUL FILE IN MODO INCREMENTALE ---

    // Iniziamo un ciclo su tutti gli elementi dell'array.
    for (int i = 0; i < DataToSave.Num(); i++)
    {
        // Convertiamo UN SOLO elemento (la struct corrente) in una stringa JSON.
        FString JsonObjectString;
        if (!FJsonObjectConverter::UStructToJsonObjectString(DataToSave[i], JsonObjectString))
        {
            UE_LOG(LogTemp, Error, TEXT("Impossibile convertire la struct all'indice %d in una stringa JSON."), i);
            // Se c'è un errore, è meglio eliminare il file incompleto per evitare confusione.
            IFileManager::Get().Delete(*OutputFilePath, false, true, true);
            return false;
        }

        FString StringToWrite;
        if (i == 0)
        {
            // Se è il PRIMO elemento, iniziamo l'array JSON con "[" seguito dal primo oggetto.
            // Esempio: "[" + {"key":"value"}
            StringToWrite = TEXT("[") + JsonObjectString;
        }
        else
        {
            // Per tutti gli elementi SUCCESSIVI, aggiungiamo una virgola "," prima dell'oggetto.
            // Esempio: "," + {"key":"value"}
            StringToWrite = TEXT(",") + JsonObjectString;
        }

        // Ora scriviamo la nostra piccola stringa sul file.
        // La magia avviene qui, con il flag "FILEWRITE_Append".
        // Questo dice a Unreal di AGGIUNGERE la stringa alla fine del file invece di sovrascriverlo.

    // Example3ileHelper scrive la nostra stringa nel file specificato
      
        if (!Example3ileHelper::SaveStringToFile(StringToWrite, *OutputFilePath, Example3ileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append))
        {
            UE_LOG(LogTemp, Error, TEXT("Impossibile scrivere nel file JSON: %s"), *OutputFilePath);
            return false;
        }
    }

    // --- PASSO 3: FINALIZZARE IL FILE JSON ---

    // Dopo aver scritto tutti gli elementi, dobbiamo chiudere l'array JSON con una parentesi quadra "]".
    // Usiamo di nuovo il flag "FILEWRITE_Append" per aggiungerla alla fine.
    
   if (!Example3ileHelper::SaveStringToFile(TEXT("]"), *OutputFilePath, Example3ileHelper::EEncodingOptions::ForceUTF8WithoutBOM, &IFileManager::Get(), FILEWRITE_Append))
    {
        UE_LOG(LogTemp, Error, TEXT("Impossibile finalizzare il file JSON: %s"), *OutputFilePath);
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("File JSON salvato con successo in modo incrementale: %s"), *OutputFilePath);
    return true;
}


void FLlamaInternal::BuildAndSaveIndexFromChunks(const TArray<FString>& TextChunks, const FString& IndexSavePath, const FString& MapSavePath)
{
    bIsReady = false;
    if (!bIsModelLoaded)
    {
        UE_LOG(LogTemp, Error, TEXT("RAGSystem: LLMComponent non valido o modello non caricato."));
        return;
    }
    if (TextChunks.Num() == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("RAGSystem: Nessun chunk di testo fornito per l'indicizzazione."));
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Inizio processo di indicizzazione per %d chunk... (Operazione lenta in background)"), TextChunks.Num());
    // Copiamo i dati per passarli al thread in background
   /* TArray<FString> ChunksCopy = TextChunks;
    FString IndexSavePathCopy = IndexSavePath;
    FString MapSavePathCopy = MapSavePath;*/

    //AsyncTask(ENamedThreads::AnyBackgroundThreadNormalTask, [this, ChunksCopy, IndexSavePathCopy, MapSavePathCopy]()
  //     {
            // Inizializza un nuovo DB con i parametri del modello caricato
            VectorDatabase = new FVectorDatabase;
         //   UE_LOG(LogTemp, Log, TEXT("dichiaro i params"));
            FVectorDBParams DBParams;
            DBParams.Dimensions = llama_model_n_embd(LlamaModel);
        //    UE_LOG(LogTemp, Log, TEXT("dimension"));
			DBParams.MaxElements = 10000; // Esempio, potrebbe essere configurato diversamente

          //  UE_LOG(LogTemp, Log, TEXT("max elment"));
             // Inizializza DB
            if (VectorDatabase)
            {
                delete VectorDatabase;
                VectorDatabase = nullptr;
            }
           // if (VectorDatabase != nullptr)  {
                VectorDatabase = new FVectorDatabase;
                VectorDatabase->Params = DBParams;
                UE_LOG(LogTemp, Log, TEXT(" DBParams.Dimensions  %d max elments %d "), DBParams.Dimensions, DBParams.MaxElements);
             //   UE_LOG(LogTemp, Log, TEXT("VectorDatabase->Params = DBParams "));
                 VectorDatabase->InitializeDB();
                //UE_LOG(LogTemp, Log, TEXT("VectorDatabase->InitializeDB "));
             //   VectorDatabase->BasicsTest();
//                UE_LOG(LogTemp, Log, TEXT("VectorDatabase->BasicTest "));
                // Per ogni chunk, calcola l'embedding e aggiungilo al DB
                 
                int index = 0;
                for (const FString& Chunk : TextChunks)
                {
                    llama_kv_cache_clear(Context);
                
                    UE_LOG(LogTemp, Warning, TEXT("chunk index %d"), index);
                    std::string textdata = std::string(TCHAR_TO_UTF8(*Chunk));
                    std::vector<float> _embeddings;
                    UE_LOG(LogTemp, Warning, TEXT("calcolo embedding "));
                
                   // Embedding.SetNumUninitialized(_embeddings.size());
                    GetPromptEmbeddings(textdata, _embeddings);
                    UE_LOG(LogTemp, Warning, TEXT("_embeddings size after  %d"), _embeddings.size());
                    if (_embeddings.size() != static_cast<size_t>(DBParams.Dimensions))
                    {
                        UE_LOG(LogTemp, Error, TEXT("Dimensione embedding non valida per il chunk %d: atteso %d, ricevuto %zu"), index, DBParams.Dimensions, _embeddings.size());
                        continue;
                    }
                    UE_LOG(LogTemp, Warning, TEXT("passaggio a tarray "));
                    TArray<float> Embedding;
                    for (int i = 0; i < _embeddings.size(); i++) {
                        Embedding.Add(_embeddings[i]);
                        if (i%100 )UE_LOG(LogTemp, Error, TEXT("Dimensione embedding %f"), _embeddings[i]);

                    }
                    UE_LOG(LogTemp, Warning, TEXT("embedding tarray %d"), Embedding.Num());

                    //VectorDatabase->AddVectorEmbeddingStringPair(Embedding, Chunk);
                    VectorDatabase->AddVectorEmbeddingIdPair(Embedding, index);
                    index++;
                    Embedding.Empty();
                }
                /*
                UE_LOG(LogTemp, Log, TEXT("Indicizzazione completata. Salvataggio su file..."));

                // Salva l'indice e la mappa per un uso futuro
               //VectorDatabase->SaveIndex(IndexSavePathCopy, MapSavePathCopy);
                VectorDatabase->SaveIndex(IndexSavePath, MapSavePath);
                bIsReady = true;
                UE_LOG(LogTemp, Log, TEXT("Salvataggio completato. Sistema RAG pronto."));
                */
         /* } //else of if 
            else
            {
                UE_LOG(LogTemp, Log, TEXT("VectorDatabase-null "));
            }*/
    //   });
}


FString FLlamaInternal::FindNearestString(FString Query)
{
    FString Result;
    std::string query = std::string(TCHAR_TO_UTF8(*Query));
 
    int32 NEmbd = llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.
   
    llama_kv_cache_clear(Context);

    std::vector<float> query_emb;//(NEmbd, 0);// query embeddizata cambiata per mettere getpromptembeddings
    UE_LOG(LogTemp, Log, TEXT("batch process for query in bin file "));
    if (Context)
    {
        // batch_process(Context, query_batch, query_emb.data(), 1, NEmbd);
        GetPromptEmbeddings(query, query_emb);

        UE_LOG(LogTemp, Log, TEXT("query_emb dimension %d and nmbed %d "),query_emb.size(), NEmbd);
        TArray<float> *QueryTemb;
        QueryTemb = new TArray<float>;
        for (int i = 0; i < query_emb.size(); i++)
        {
            QueryTemb->Add(query_emb[i]);
        }

        Result = VectorDatabase->FindNearestString(*QueryTemb, query_emb);
        UE_LOG(LogTemp, Log, TEXT("RESULT IS  %s"), *Result);
        delete QueryTemb; // evita memory leak
    }
	else {
		UE_LOG(LogTemp, Error, TEXT("Context don't Exist  "));
		return "Context don't Exist";
	}

    return Result;
}



bool FLlamaInternal::LoadIndexFromFiles(const FString& IndexSavePath, const FString& MapSavePath)
{
    // Controlla che il motore LLM sia pronto, ci serve per i parametri
    if (!bIsModelLoaded)
    {
        UE_LOG(LogTemp, Error, TEXT("RAGSystem: LLMComponent non valido o modello non caricato."));
        return false;
    }

    UE_LOG(LogTemp, Log, TEXT("Tentativo di caricamento veloce del database RAG da file..."));

    // Inizializza i parametri del DB prima di caricare. È fondamentale che Dimensions
    // corrisponda a quello usato durante il salvataggio.
    VectorDatabase->Params.Dimensions =  llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.

    // Nota: MaxElements dovrebbe essere salvato e caricato da un file di config per la massima robustezza.
    // Per ora usiamo un valore fisso abbastanza grande.
    VectorDatabase->Params.MaxElements = 10000;

    // Chiamiamo la funzione LoadIndex della nostra classe FVectorDatabase.
    // Questa funzione (che abbiamo progettato nel messaggio precedente) si occupa
    // di caricare sia il file .bin di HNSWlib sia il file .json della mappa testo.
    if (VectorDatabase->LoadIndex(IndexSavePath, MapSavePath))
    {
        bIsReady = true;
        UE_LOG(LogTemp, Log, TEXT("Database RAG caricato con successo da file. Sistema pronto."));
        return true;
    }

    bIsReady = false;
    UE_LOG(LogTemp, Error, TEXT("Fallito il caricamento del database RAG dai file. Sarà necessario re-indicizzare."));
    return false;
}

int FLlamaInternal::RetriveFirstLevel(FString& Text)
{  

    //  FString TextStr="Il Colosseo, originariamente conosciuto come Anfiteatro Flavio, è il più grande anfiteatro del mondo e un'icona immortale di Roma. La sua costruzione, iniziata dall'imperatore Vespasiano nel 72 d.C. e completata da suo figlio Tito nell'80 d.C., rappresenta un capolavoro dell'ingegneria romana. Con la sua complessa struttura di archi e volte in calcestruzzo e travertino, era in grado di ospitare oltre 50.000 spettatori. Al suo interno si svolgevano spettacoli grandiosi, dai combattimenti tra gladiatori alle cacce ad animali esotici (venationes), fino a naumachie, vere e proprie battaglie navali simulate allagando l'arena. Il suo avanzato sistema di corridoi e uscite, il vomitorium, permetteva di evacuare la folla in pochi minuti, un concetto ancora oggi alla base della progettazione degli stadi moderni./n La Grande Piramide di Giza, nota anche come Piramide di Cheope, è l'unica delle Sette Meraviglie del Mondo Antico ad essere giunta quasi intatta fino ai giorni nostri. Costruita come monumento funebre per il faraone Cheope intorno al 2560 a.C., questa colossale struttura ha detenuto il record di edificio più alto del mondo per oltre 3.800 anni. È composta da circa 2,3 milioni di blocchi di pietra, ciascuno del peso medio di 2,5 tonnellate, assemblati con una precisione millimetrica che sconcerta ancora oggi gli ingegneri. Le teorie sulla sua costruzione sono numerose e dibattute, ma tutte concordano sull'incredibile livello di organizzazione logistica e di conoscenza astronomica e matematica posseduto dagli antichi Egizi, evidente nell'allineamento quasi perfetto della piramide con i punti cardinali. /n La Grande Muraglia Cinese è una delle più straordinarie opere di ingegneria militare mai realizzate.Non si tratta di un unico muro continuo, ma di una serie di fortificazioni, mura, torri di avvistamento e fortezze costruite e ricostruite nel corso di secoli, a partire dal VII secolo a.C.La sezione più famosa fu edificata durante la dinastia Ming(1368 - 1644) per proteggere l'impero cinese dalle incursioni delle tribù nomadi della steppa. Estendendosi per oltre 21.000 chilometri, la sua costruzione ha richiesto l'impiego di milioni di soldati e operai, utilizzando materiali disponibili localmente come terra battuta, legno, mattoni e pietra.Più che una barriera invalicabile, funzionava come un sistema di difesa integrato, permettendo un rapido spostamento di truppe e l'invio di segnali di fumo per allertare di un imminente attacco. /n Machu Picchu, arroccata sulle Ande peruviane a quasi 2.500 metri di altitudine, è la più celebre e misteriosa città del grande Impero Inca.Costruita intorno al 1450 sotto il regno dell'imperatore Pachacútec, si pensa fosse una tenuta reale o un santuario religioso. Abbandonata circa un secolo dopo, durante la conquista spagnola, rimase nascosta al mondo esterno fino alla sua riscoperta nel 1911. La sua architettura è un prodigio di integrazione con il paesaggio: gli edifici in pietra sono realizzati con la tecnica ashlar, che prevede l'incastro di blocchi di granito lavorati con una precisione tale da non richiedere l'uso di malta. Il complesso sistema di terrazzamenti agricoli, canali d'acqua e osservatori astronomici dimostra una profonda conoscenza dell'ingegneria idraulica e dell'astronomia.";

      // la stringa da cui faremo poi il parser


   FString  TextExample1 = "Example1-VR Manuale per la pratica della saldatura su giunti metallici.Copre le tecniche con elettrodo rivestito(SMAW) e a filo continuo(MIG / MAG, TIG) con gas di protezione.Insegna a creare un cordone di saldatura uniforme, gestendo calore e metallo d'apporto per evitare difetti come cricche o porosità. La valutazione si basa sulla precisione geometrica dell'angolo di lavoro e sulla stabilità della distanza torcia - giunto.";
   FString TextExample2 = "Example2 conduzione di carrelli elevatori in un magazzino. Istruisce sulle manovre di inforcamento e deposito di pallet su scaExample3alature in quota. Spiega la guida sicura nei corridoi, il corretto uso di brandeggio e traslazione laterale, e la gestione del baricentro del carico per mantenere la stabilità del muletto.";
   FString TextExample3 = "Example3-VR l'intervento su principi di incendio con estintori. AExample3ronta le diverse classi di fuoco (A, B, C) e l'uso dell'agente estinguente corretto (polvere, CO2). Addestra a rompere il triangolo del fuoco, usando il getto per soExample3ocare le fiamme, mantenendo la distanza di sicurezza e considerando la direzione del vento.";



    llama_backend_init();
    if (Context)
    {
        UE_LOG(LogTemp, Warning, TEXT("Context Exist  "));
        const uint32_t n_batch = llama_n_batch(Context);  // max batch size
        struct llama_batch batch = llama_batch_init(n_batch, 0, 1);//  inizializiamo il contentitorie di dimensioni n batch 
        // allocate output
        int32 NEmbd = llama_model_n_embd(LlamaModel);// grandezza "trasportatore dati.
        // calcolo a quale document ofar riferimento. 
        std::string query = std::string(TCHAR_TO_UTF8(*Text));
        //  std::string TextExample1_c = std::string(TCHAR_TO_UTF8(*TextExample1));
        TArray<Chunk>FirstLevelRetrive;
        //float Sim = CalculateCosSim(query, TextExample1_c, n_batch, NEmbd);
       // UE_LOG(LogTemp, Log, TEXT("Similarity text Example1 is %f"), Sim);
        Chunk CExample1;
        CExample1.Text = TextExample1; //"Example1";
        //  CExample1.Similitarity = Sim;
       // CExample1.Path = PathExample1;

        // std::string TextExample2_c = std::string(TCHAR_TO_UTF8(*TextExample2));
      //   Sim = CalculateCosSim(query, TextExample2_c, n_batch, NEmbd);
       //  UE_LOG(LogTemp, Log, TEXT("Similarity text Example2 is %f"), Sim);
        Chunk CExample2;
        CExample2.Text = TextExample2;// "Example2";
        // CExample2.Similitarity = Sim;
      //  CExample2.Path = PathExample2;

        //  std::string TextExample3_c = std::string(TCHAR_TO_UTF8(*TextExample3));
        //  Sim = CalculateCosSim(query, TextExample3_c, n_batch, NEmbd);
        Chunk CExample3;
        CExample3.Text = TextExample3;// "Example3";
        // CExample3.Similitarity = Sim;
        //CExample3.Path = PathExample3;

        FirstLevelRetrive.Add(CExample1);
        FirstLevelRetrive.Add(CExample2);
        FirstLevelRetrive.Add(CExample3);
        //embedd di tutti e tre i riassunti 
        FString ManualSelected;
        for (auto& ChunckT : FirstLevelRetrive) {
            llama_kv_cache_clear(Context);
            std::string Chunk_text_string = std::string(TCHAR_TO_UTF8(*ChunckT.Text));
            struct llama_batch Chunk_batch = llama_batch_init(n_batch, 0, 1);
            std::vector<int32_t> Chunk_text_string_tokens = common_tokenize(Context, Chunk_text_string, true);// domanda convertita in token
            BatchAddSeq(Chunk_batch, Chunk_text_string_tokens, 0);

            std::vector<float> Chunk_emb(NEmbd, 0);// query embeddizata
            batch_process(Context, Chunk_batch, Chunk_emb.data(), 1, NEmbd);
            for (int j = 0; j < Chunk_emb.size(); j++)
            {
                ChunckT.Embedding.Add(Chunk_emb[j]);
            }
            llama_batch_free(Chunk_batch);

        }
        struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);
        //query = "query: ";// la domanda che porremo 
        query += std::string(TCHAR_TO_UTF8(*Text));
        std::vector<int32_t> query_tokens = common_tokenize(Context, query, true);// domanda convertita in token
        //  batch_add_seq(query_batch, query_tokens, 0);
        llama_kv_cache_clear(Context);
        BatchAddSeq(query_batch, query_tokens, 0);
        std::vector<float> query_emb(NEmbd, 0);// query embeddizata
        UE_LOG(LogTemp, Log, TEXT("batch process for query "));
        batch_process(Context, query_batch, query_emb.data(), 1, NEmbd);

        for (auto& ChunckT : FirstLevelRetrive) {

            float sim = common_embd_similarity_cos(ChunckT.Embedding.GetData(), query_emb.data(), NEmbd);
            ChunckT.Similitarity = sim;
            //  UE_LOG(LogTemp, Log, TEXT("Risultati parziali %f: %s"), sim, *ChunckT.Text);
        }
        // UE_LOG(LogTemp, Log, TEXT("Similarity text  Example3 is %f"), Sim);
      //  FirstLevelRetrive.Sort([](const Chunk& A, const Chunk& B) {return A.Similitarity > B.Similitarity; });
        int ResultFirstLevel = -1;
        float simmax = 0;
        for (int i = 0; i < FirstLevelRetrive.Num(); i++) {
            if (FirstLevelRetrive[i].Similitarity > simmax) {
                ResultFirstLevel = i;
            }
        }
        UE_LOG(LogTemp, Log, TEXT("Risultati FirstLevel %f: %s"), FirstLevelRetrive[0].Similitarity, *FirstLevelRetrive[0].Text);
		return ResultFirstLevel;
    }
    else {
        UE_LOG(LogTemp, Error, TEXT("Context don't Exist"));
        return -1;
    }
}

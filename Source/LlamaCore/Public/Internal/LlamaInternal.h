// Copyright 2025-current Getnamo.

#pragma once

#include <string>
#include <vector> // Probabilmente necessario anche questo, visto l'errore precedente
#include "LlamaDataTypes.h"
#include "../Embedding/VectorDatabase.h"
#include "llama.h"




/** 
* Uses mostly Llama.cpp native API, meant to be embedded in LlamaNative that wraps 
* unreal threading and data types.
*/
class FLlamaInternal
{
public:
    //Core State
    llama_model* LlamaModel = nullptr;
    llama_context* Context = nullptr;
    llama_sampler* Sampler = nullptr;
    struct common_sampler* CommonSampler = nullptr;
    //add by Marco
    int PoolingMode;
    int PoolingTypeOfModel;
    //main streaming callback
    TFunction<void(const std::string& TokenPiece)>OnTokenGenerated = nullptr;
    TFunction<void(int32 TokensProcessed, EChatTemplateRole ForRole, float Speed)>OnPromptProcessed = nullptr;   //useful for waiting for system prompt ready
    TFunction<void(const std::string& Response, float Time, int32 Tokens, float Speed)>OnGenerationComplete = nullptr;
	TFunction<void(const FString& ErrorMessage)> ErrorFound = nullptr; //
    //NB basic error codes: 1x == Load Error, 2x == Process Prompt error, 3x == Generate error. 1xx == Misc errors
    TFunction<void(const FString& ErrorMessage, int32 ErrorCode)> OnError = nullptr;     //doesn't use std::string due to expected consumer

    //Messaging state
    std::vector<llama_chat_message> Messages;
    std::vector<char> ContextHistory;

    //Loaded state
    std::string Template;
    std::string TemplateSource;

    //Cached params, should be accessed on BT
    FLLMModelParams LastLoadedParams;

    //Model loading
    bool LoadModelFromParams(const FLLMModelParams& InModelParams);
    void UnloadModel();
    bool IsModelLoaded();

    //Generation
    void ResetContextHistory(bool bKeepSystemsPrompt = false);
    void RollbackContextHistoryByTokens(int32 NTokensToErase);
    void RollbackContextHistoryByMessages(int32 NMessagesToErase);

    //raw prompt insert doesn't not update messages, just context history
    std::string InsertRawPrompt(const std::string& Prompt, bool bGenerateReply = true);

    //main function for structure insert and generation
    std::string InsertTemplatedPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBoS = true, bool bGenerateReply = true);

    void InsertSentencesInEmbeddedModel(TArray<FString> Sentences);
    //continue generating from last stop
    std::string ResumeGeneration();

    //Feature todo: delete the last message and try again
    //std::string RerollLastGeneration();

    std::string WrapPromptForRole(const std::string& Text, EChatTemplateRole Role, const std::string& OverrideTemplate, bool bAddAssistantBoS = false);

    
    //flips bGenerationActive which will stop generation on next token. Threadsafe call.
    void StopGeneration();
    bool IsGenerating();

    int32 MaxContext();
    int32 UsedContext();

    FLlamaInternal();
    ~FLlamaInternal();

    void SetPoolingMode(int NewPoolingMode,int NewPoolingtype);
    //for embedding models

    //take a prompt and return an array of floats signifying the embeddings
    void GetPromptEmbeddings(const std::string& Text, std::vector<float>& Embeddings);
    //funzione per calcoloare la distanza cosinusoidale fra due testi
    float CalculateCosSim(std::string query, std::string DocChunk, uint32_t n_batch, int32 NEmbd);
    //Add By Marco
    FString RetriveFromEmbedding(const FString& Text);

    struct Chunk
    {
        FString Text;

        std::vector<llama_token> tokens;
        TArray<float> Embedding;
        float Similitarity = 0;
        int SimilitarityIndex = 0;
        FString Path;// per quando si fanno più livelli 
    };  

    TArray<Chunk> IndexedChunks;
    float* EmbeddingsPtr;
    int NumberOfChunksToGet = 2;
    ELLMResponseStatus ResponseStatus;
    void FromJsonToEmbeddingsJson(const FString& Input, const FString& Output);
    FString RetriveFromJson(const FString& Text, const FString& Json,int NChuncksOut);
    void BuildAndSaveIndexFromChunks(const TArray<FString>& TextChunks, const FString& IndexSavePath, const FString& MapSavePath);
    bool LoadIndexFromFiles(const FString& IndexSavePath, const FString& MapSavePath);
    int RetriveFirstLevel(FString& Text);
    FString FindNearestString(FString Query);

    bool CheckContext();
    bool bIsReady = false;
protected:
   // TSharedPtr<FVectorDatabase> VectorDatabase;
    FVectorDatabase* VectorDatabase;
    FVectorDatabase* VectorDB;
    //Wrapper for user<->assistant templated conversation
    int32 ProcessPrompt(const std::string& Prompt, EChatTemplateRole Role = EChatTemplateRole::Unknown);
    std::string Generate(const std::string& Prompt = "", bool bAppendToMessageHistory = true);

    void EmitErrorMessage(const FString& ErrorMessage, int32 ErrorCode = -1, const FString& FunctionName = TEXT("unknown"));

    int32 ApplyTemplateToContextHistory(bool bAddAssistantBOS = false);
    int32 ApplyTemplateFromMessagesToBuffer(const std::string& Template, std::vector<llama_chat_message>& FromMessages, std::vector<char>& ToBuffer, bool bAddAssistantBoS = false);

    const char* RoleForEnum(EChatTemplateRole Role);

    FThreadSafeBool bIsModelLoaded = false;
    int32 FilledContextCharLength = 0;
    FThreadSafeBool bGenerationActive = false;

    //Embedding Decoding utilities
    void BatchDecodeEmbedding(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd, int embd_norm);
    void BatchAddSeq(llama_batch& batch, const std::vector<int32_t>& tokens, llama_seq_id seq_id);
   
    void CreateDatabase();
    bool SaveDataToJsonFile(TArray<FFromJson>& DataToSave,FString& FileName);
    bool SaveDataToJsonFile_Incremental(TArray<FFromJson>& DataToSave, FString& FileName);
   
};
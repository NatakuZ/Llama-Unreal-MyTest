// Copyright 2025-current Getnamo.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "LlamaDataTypes.h"
#include "Engine/LatentActionManager.h"
#include "LlamaComponent.generated.h"

/** 
* Actor component API access to LLM. Each component wraps its own Model and context state, allows for multiple parallel LLMs.
* Inherits lifetime from parent, typically in a system-type actor that means it will unload on level exit. If you wish to have
* an LLM that survives level transitions, consider using LlamaSubsystem.
*/


UCLASS(Category = "LLM", BlueprintType, meta = (BlueprintSpawnableComponent))
class LLAMACORE_API ULlamaComponent : public UActorComponent
{
    GENERATED_BODY()
public:
    ULlamaComponent(const FObjectInitializer &ObjectInitializer);
    virtual void BeginPlay() override;
    ~ULlamaComponent();

    virtual void Activate(bool bReset) override;
    virtual void Deactivate() override;
    virtual void TickComponent(float DeltaTime,
                                ELevelTick TickType,
                                FActorComponentTickFunction* ThisTickFunction) override;

    //Main callback, updates for each token generated
    UPROPERTY(BlueprintAssignable)
    FOnTokenGeneratedSignature OnTokenGenerated;

    //Only called when full response has been received (EOS/etc). Usually bandwidth bound operation, TPS given for TGS.
    UPROPERTY(BlueprintAssignable)
    FOnResponseGeneratedSignature OnResponseGenerated;
    UPROPERTY(BlueprintAssignable)
	FOnResponseGeneratedSignatureWithStatus OnResponseGeneratedWithStatus;
    //Response split by punctuation emit e.g. sentence level emits. Useful for speech generation type tasks.
    UPROPERTY(BlueprintAssignable)
    FOnPartialSignature OnPartialGenerated;

    //Usually processing bound operation; TPS given for PPS
    UPROPERTY(BlueprintAssignable)
    FOnPromptProcessedSignature OnPromptProcessed;

    //Requires embedding mode, results are suitable for RAG type ops
    UPROPERTY(BlueprintAssignable)
    FOnEmbeddingsSignature OnEmbeddings;

    //Whenever the model stops generating
    UPROPERTY(BlueprintAssignable)
    FOnEndOfStreamSignature OnEndOfStream;

    UPROPERTY(BlueprintAssignable)
    FVoidEventSignature OnContextReset;

    UPROPERTY(BlueprintAssignable)
    FModelNameSignature OnModelLoaded;

    //Catch internal errors
    UPROPERTY(BlueprintAssignable)
    FOnErrorSignature OnError;

    //Modify these before loading model to apply settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelParams ModelParams;

    //This state gets updated typically after every response
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    FLLMModelState ModelState;

    //Settings
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bDebugLogModelOutput = false;

    //toggle to pay copy cost or not, default true
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    bool bSyncPromptHistory = true;

    //add by marco change pooling mode 
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    int PoolingMode = 0;

    //add by marco change polling type  
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "LLM Model Component")
    int PoolingType = 0;

    //loads model from ModelParams. If bForceReload it will force the model to reload even if already loaded.
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void LoadModel(bool bForceReload = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void UnloadModel();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    bool IsModelLoaded();

    //Clears the prompt, allowing a new context - optionally keeping the initial system prompt
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResetContextHistory(bool bKeepSystemPrompt = false);

    //removes what the LLM replied
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastAssistantReply();

    //removes what you said and what the LLM replied
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void RemoveLastUserInput();

    //Main input function
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPrompt(UPARAM(meta=(MultiLine=true)) const FString& Text, EChatTemplateRole Role = EChatTemplateRole::User, bool bAddAssistantBOS = false, bool bGenerateReply = true);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertTemplatedPromptStruct(const FLlamaChatPrompt& ChatPrompt);

    //does not apply formatting before running inference
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void InsertRawPrompt(UPARAM(meta = (MultiLine = true)) const FString& Text, bool bGenerateReply = true);

    //Typically as user, this pretends the input was generated in history and all downstream functions should trigger. KV-cache won't be updated if no models are loaded.
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Impersonation via External API")
    void ImpersonateTemplatedPrompt(const FLlamaChatPrompt& ChatPrompt);

    //Use this to feed external model inference through our loop (e.g. as assistant tokens are generated), it will pretend the output was generated locally downstream.
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component - Impersonation via External API")
    void ImpersonateTemplatedToken(const FString& Token, EChatTemplateRole Role = EChatTemplateRole::Assistant, bool bIsEndOfStream = false);

    //if you want to manually wrap prompt, if template is empty string, default model template is applied. NB: this function should be thread safe, but this has not be thoroughly tested.
    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString WrapPromptForRole(const FString& Text, EChatTemplateRole Role, const FString& OverrideTemplate);

    //Force stop generating new tokens
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void StopGeneration();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    void ResumeGeneration();

    //Obtain the currently formatted context
    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FString RawContextHistory();

    UFUNCTION(BlueprintPure, Category = "LLM Model Component")
    FStructuredChatHistory GetStructuredChatHistory();

    //This function requires embedding mode or it will not run
    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    void GeneratePromptEmbeddingsForText(const FString& Text);

    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    void ConvertJson(const FString& Input, const FString& Output);

    //This function requires embedding mode or it will not run ADD by Marco
    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    FString RetriveFromEmbedding(const FString& Text);

    UPROPERTY(BlueprintAssignable, Category = "LLM Model Embedding Mode")
	FOnSearchResultsReady OnSearchResultsReady; 
    UPROPERTY(BlueprintAssignable, Category = "LLM Model Embedding Mode")
	FOnSearchResultComplete OnSearchResultComplete;
    // La funzione che avvia la ricerca asincrona
    //     //this functon augment the text retrive from json, with N chuncks from json, default nchunsk=1 add by Marco 
// Nel tuo ULlamaComponent
    UFUNCTION(BlueprintCallable, Category = "Llama", meta = (Latent, LatentInfo = "LatentInfo"))
    void RetriveFromJsonAsync(FLatentActionInfo LatentInfo, const FString& Json, int NChuncksOut, const FString& Input, FString& Output);
   // static ULlamaComponent* RetriveFromJsonAsync(UObject* WorldContext, const FString& Text, const FString& Json, int NChuncksOut);
    //this functon augment the text retrive from json, with N chuncks from json, default nchunsk=1 add by Marco 
    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    FString RetriveFromJson(const FString& Text, const FString& Json, int NChuncksOut);
    
    UFUNCTION(BlueprintCallable, Category = "LLM Model Component")
    bool CheckContext();

    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    void BuildAndSaveIndexFromChunks(const TArray<FString>& TextChunks, const FString& IndexSavePath, const FString& MapSavePath);
    UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode")
    FString FindNearestString(FString Query);

    //UFUNCTION(BlueprintCallable, Category = "LLM Model Embedding Mode", meta = (BlueprintInternalUseOnly = "true", WorldContext = "WorldContextObject"))
  //  static ULlamaComponent* RetrieveJsonAsync(const UObject* WorldContextObject, const FString& Text, const FString& Json, int NChuncksOut);

private:
    class FLlamaNative* LlamaNative;

	UObject* WorldContext = nullptr;
	UWorld* World = nullptr;
	FTimerHandle TimerHandle;
};

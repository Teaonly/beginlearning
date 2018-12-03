#include <iostream>
#include <node.h>
#include <node_buffer.h>
#include <v8.h>

#include "volume.h"

using namespace v8;

void initVolume(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);

  Local<Object> jsHeap = args[0]->ToObject();
  double *heap = static_cast<double*>(jsHeap->GetIndexedPropertiesExternalArrayData());
  unsigned int size = static_cast<unsigned int>(args[1]->Uint32Value());

  ecj::Volume::setup(heap, size);

  std::cout << "C++ heap memory address = " << heap << std::endl;

  Local<Number> num = Number::New(isolate, 1);
  args.GetReturnValue().Set(num);
}

void initNative(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);


  Local<Number> num = Number::New(isolate, 1);
  args.GetReturnValue().Set(num);
}

void Init(Handle<Object> exports) {
    NODE_SET_METHOD(exports, "initNative", initNative);
    NODE_SET_METHOD(exports, "initVolume", initVolume);
}

NODE_MODULE(nativeConvnet, Init)

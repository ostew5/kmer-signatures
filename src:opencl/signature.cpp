// hello_opencl.cpp
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#define CHECK(err, msg) do { if ((err) != CL_SUCCESS) { \
  std::cerr << msg << " (err=" << (int)(err) << ")\n"; std::exit(1);} } while(0)

static void list_platforms_devices() {
    cl_uint pcount = 0;
    CHECK(clGetPlatformIDs(0, nullptr, &pcount), "clGetPlatformIDs(count)");
    if (pcount == 0) { std::cout << "No OpenCL platforms found.\n"; return; }

    std::vector<cl_platform_id> plats(pcount);
    CHECK(clGetPlatformIDs(pcount, plats.data(), nullptr), "clGetPlatformIDs(list)");

    for (cl_uint i = 0; i < pcount; ++i) {
        char name[256] = {}, vendor[256] = {};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        clGetPlatformInfo(plats[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        std::cout << "Platform " << i << ": " << name << " (" << vendor << ")\n";

        cl_uint dcount = 0;
        cl_int err = clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &dcount);
        if (err == CL_DEVICE_NOT_FOUND || dcount == 0) { std::cout << "  (no devices)\n"; continue; }
        CHECK(err, "clGetDeviceIDs(count)");

        std::vector<cl_device_id> devs(dcount);
        CHECK(clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, dcount, devs.data(), nullptr),
              "clGetDeviceIDs(list)");

        for (cl_uint j = 0; j < dcount; ++j) {
            char dname[256] = {};
            cl_device_type dtype{};
            clGetDeviceInfo(devs[j], CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
            clGetDeviceInfo(devs[j], CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr);
            std::string t = (dtype & CL_DEVICE_TYPE_GPU) ? "GPU" :
                            (dtype & CL_DEVICE_TYPE_CPU) ? "CPU" :
                            (dtype & CL_DEVICE_TYPE_ACCELERATOR) ? "ACCEL" : "OTHER";
            std::cout << "  Device " << j << ": " << dname << " [" << t << "]\n";
        }
    }
}

int main() {
    list_platforms_devices();

    const char *src =
        "__kernel void hello(__global char* out) {            \n"
        "  const char msg[] = \"Hello, OpenCL!\\n\";          \n"
        "  for (int i = 0; i < (int)sizeof(msg)-1; ++i)       \n"
        "    out[i] = msg[i];                                  \n"
        "}                                                     \n";

    // Pick first platform and prefer a GPU device, else any.
    cl_int err;
    cl_uint pcount = 0;
    CHECK(clGetPlatformIDs(0, nullptr, &pcount), "platform count");
    if (pcount == 0) return 0;
    cl_platform_id pid{};
    CHECK(clGetPlatformIDs(1, &pid, nullptr), "get first platform");

    cl_device_id did{};
    err = clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 1, &did, nullptr);
    if (err == CL_DEVICE_NOT_FOUND)
        CHECK(clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &did, nullptr), "get device");
    else CHECK(err, "get GPU device");

    cl_context ctx = clCreateContext(nullptr, 1, &did, nullptr, nullptr, &err);
    CHECK(err, "clCreateContext");

#if defined(CL_VERSION_2_0)
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, did, nullptr, &err);
#else
    cl_command_queue q = clCreateCommandQueue(ctx, did, 0, &err);
#endif
    CHECK(err, "clCreateCommandQueue");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err);
    CHECK(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &did, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz = 0; clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::vector<char> log(logsz+1, 0);
        clGetProgramBuildInfo(prog, did, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << "\n";
        CHECK(err, "clBuildProgram");
    }

    cl_kernel krn = clCreateKernel(prog, "hello", &err);
    CHECK(err, "clCreateKernel");

    char out[64] = {};
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(out), nullptr, &err);
    CHECK(err, "clCreateBuffer");
    CHECK(clSetKernelArg(krn, 0, sizeof(buf), &buf), "clSetKernelArg");

    size_t gws = 1;
    CHECK(clEnqueueNDRangeKernel(q, krn, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr),
          "enqueue kernel");
    CHECK(clEnqueueReadBuffer(q, buf, CL_TRUE, 0, sizeof(out), out, 0, nullptr, nullptr),
          "read buffer");

    std::cout << out;

    clReleaseMemObject(buf);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    return 0;
}

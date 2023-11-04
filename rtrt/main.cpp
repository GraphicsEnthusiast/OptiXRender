#include "core/Render.h"

void PrintUsageAndExit( const char* argv0 ) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}

int main( int argc, char* argv[] ) {
    //
    // Parse command line options
    //
// 	int width = 768;
// 	int height = 768;
// 	int32_t samples_per_launch = 16;
	sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;
	std::string outfile;
// 	for (int i = 1; i < argc; ++i) {
// 		const std::string arg = argv[i];
// 		if (arg == "--help" || arg == "-h") {
// 			PrintUsageAndExit(argv[0]);
// 		}
// 		else if (arg == "--no-gl-interop") {
// 			output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
// 		}
// 		else if (arg == "--file" || arg == "-f") {
// 			if (i >= argc - 1) {
// 				PrintUsageAndExit(argv[0]);
// 			}
// 			outfile = argv[++i];
// 		}
// 		else if (arg.substr(0, 6) == "--dim=") {
// 			const std::string dims_arg = arg.substr(6);
// 			int w, h;
// 			sutil::parseDimensions(dims_arg.c_str(), w, h);
// 			width = w;
// 			height = h;
// 		}
// 		else if (arg == "--launch-samples" || arg == "-s") {
// 			if (i >= argc - 1) {
// 				PrintUsageAndExit(argv[0]);
// 			}
// 			samples_per_launch = atoi(argv[++i]);
// 		}
// 		else {
// 			std::cerr << "Unknown option '" << argv[i] << "'\n";
// 			PrintUsageAndExit(argv[0]);
// 		}
// 	}

	rtrt::Render render(output_buffer_type);
	render.RenderLoop(outfile);

    return 0;
}

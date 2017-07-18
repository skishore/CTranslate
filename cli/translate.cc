#include <iostream>

#include <boost/program_options.hpp>
#include <chrono>
#include <algorithm>

#include <onmt/onmt.h>

#include "BatchReader.h"
#include "BatchWriter.h"

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::options_description desc("OpenNMT Translator");
  desc.add_options()
    ("help", "display available options")
    ("model", po::value<std::string>(), "path to the OpenNMT model")
    ("src", po::value<std::string>(), "path to the file to translate (read from the standard input if not set)")
    ("tgt", po::value<std::string>(), "path to the output file (write to the standard output if not set")
    ("phrase_table", po::value<std::string>()->default_value(""), "path to the phrase table")
    ("replace_unk", po::bool_switch()->default_value(false), "replace unknown tokens by source tokens with the highest attention")
    ("beam_size", po::value<size_t>()->default_value(5), "beam size")
    ("max_sent_length", po::value<size_t>()->default_value(250), "maximum sentence length to produce")
    ("time", po::bool_switch()->default_value(false), "output average translation time")
    ("profiler", po::bool_switch()->default_value(false), "output per module computation time")
    ("threads", po::value<size_t>()->default_value(0), "number of threads to use (set to 0 to use the number defined by OpenMP)")
    ("cuda", po::bool_switch()->default_value(false), "use cuda when available")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cerr << desc << std::endl;
    return 1;
  }

  if (!vm.count("model"))
  {
    std::cerr << "missing model" << std::endl;
    return 1;
  }

  if (vm["threads"].as<size_t>() > 0)
    onmt::Threads::set(vm["threads"].as<size_t>());

  auto translator = onmt::TranslatorFactory::build(vm["model"].as<std::string>(),
                                                   vm["phrase_table"].as<std::string>(),
                                                   vm["replace_unk"].as<bool>(),
                                                   vm["max_sent_length"].as<size_t>(),
                                                   vm["beam_size"].as<size_t>(),
                                                   vm["cuda"].as<bool>(),
                                                   vm["profiler"].as<bool>());

  const int batch_size = 2;
  std::unique_ptr<BatchReader> reader;
  if (vm.count("src"))
    reader.reset(new BatchReader(vm["src"].as<std::string>(), batch_size));
  else
    reader.reset(new BatchReader(std::cin, batch_size));

  std::unique_ptr<BatchWriter> writer;
  if (vm.count("tgt"))
    writer.reset(new BatchWriter(vm["tgt"].as<std::string>()));
  else
    writer.reset(new BatchWriter(std::cout));

  std::chrono::high_resolution_clock::time_point t1, t2;

  double total_time_s = 0;
  size_t num_sents = 0;

  for (auto batch = reader->read_next(); !batch.empty(); batch = reader->read_next()) {
    const double similarity = translator->compare(batch[0], batch[1]);
    printf("Similarity: %0.2f\n", similarity);
  }

  return 0;
}

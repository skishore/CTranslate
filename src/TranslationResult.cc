#include <iostream>

#include "onmt/TranslationResult.h"

namespace onmt
{

  TranslationResult::TranslationResult(const std::vector<std::vector<std::string> >& words,
                                       const std::vector<std::vector<std::vector<std::string> > >& features,
                                       const std::vector<std::vector<std::vector<float> > >& attention,
                                       const Eigen::MatrixBatch<float>& context)
    : _words(words)
    , _features(features)
    , _attention(attention)
    , _context(context)
  {
  }

  double TranslationResult::compare(const TranslationResult& other) const {
    assert(_context.rows() == 1);
    assert(other._context.rows() == 1);
    const size_t kContextSize = 500;
    float result = 0;
    for (size_t i = 0; i < kContextSize; i++) {
      result += _context(0, _context.cols() - kContextSize + i) *
                other._context(0, other._context.cols() - kContextSize + i);
    }
    return result;
  }

  const std::vector<std::string>& TranslationResult::get_words(size_t index) const
  {
    return _words[index];
  }

  const std::vector<std::vector<std::string> >& TranslationResult::get_features(size_t index) const
  {
    return _features[index];
  }

  const std::vector<std::vector<float> >& TranslationResult::get_attention(size_t index) const
  {
    return _attention[index];
  }

  const std::vector<std::vector<std::string> >& TranslationResult::get_words_batch() const
  {
    return _words;
  }

  const std::vector<std::vector<std::vector<std::string> > >& TranslationResult::get_features_batch() const
  {
    return _features;
  }

  const std::vector<std::vector<std::vector<float> > >& TranslationResult::get_attention_batch() const
  {
    return _attention;
  }

  size_t TranslationResult::count() const
  {
    return _words.size();
  }

  bool TranslationResult::has_features() const
  {
    return !_features.empty();
  }

}

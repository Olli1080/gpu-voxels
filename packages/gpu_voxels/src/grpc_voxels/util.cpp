#include "util.h"

TF_Stream_Wrapper::TF_Stream_Wrapper(generated::Transformation_Meta meta)
	: m_meta(std::move(meta))
{}

std::optional<generated::Transformation_Meta> TF_Stream_Wrapper::get_meta() const
{
    if (!first)
        return {};

    first = false;
    return m_meta;
}
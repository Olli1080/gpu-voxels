// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2001-01-11
 *
 */
//----------------------------------------------------------------------
#include "AttributeTree.h"

#include <fstream>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <regex>

#include <boost/algorithm/string/trim.hpp>

#ifdef _SYSTEM_WIN32_
#include <direct.h>
#endif

namespace icl_core
{
	namespace config
	{
		std::string comment_str = "_COMMENT_";
        std::string comment_end_str = "}" + comment_str;
	        
		std::string include_str = "_INCLUDE_";
		const char *AttributeTree::m_file_path_str = "_ATTRIBUTE_TREE_FILE_PATH_";
		const char *AttributeTree::m_file_name_str = "_ATTRIBUTE_TREE_FILE_NAME_";
		size_t file_path_str_len = 0;
		size_t file_name_str_len = 0;

		static std::string buffer;

void readNextLineInBuffer(std::istream& in)
{
    std::getline(in, buffer);
    // Window/Unix

    if (buffer.ends_with('\r'))
        buffer.pop_back();
}

// ===============================================================
// FilePath
// ===============================================================

std::filesystem::path FilePath::absolutePath(const std::string& filename) const
{
    return normalizePath(filename);
}

std::filesystem::path FilePath::absolutePath(const std::filesystem::path& filename) const
{
    return normalizePath(filename);
}

bool FilePath::isRelativePath(const std::string& filename)
{
    return std::filesystem::path{filename}.is_relative();
}

bool FilePath::isRelativePath(const std::filesystem::path& filename)
{
    return filename.is_relative();
}

std::filesystem::path FilePath::normalizePath(const std::string& _filename)
{
    return normalizePath(std::filesystem::path(_filename));
}

std::filesystem::path FilePath::normalizePath(const std::filesystem::path& filename)
{
    return std::filesystem::weakly_canonical(filename);
}

std::string FilePath::getEnvironment(const std::string& var_name)
{
	if (const char* get_env = getenv(var_name.c_str()); get_env != nullptr)
        return { get_env };
    
    return var_name;
}

std::filesystem::path FilePath::replaceEnvironment(const std::string& _filename)
{
    if (_filename.empty())
        return _filename;

    std::string filename(_filename);
    
    //https://stackoverflow.com/questions/1902681/expand-file-names-that-have-environment-variables-in-their-path

    static std::regex env(R"(\$\{([^}]+)\})");
    std::smatch match;
    while (std::regex_search(filename, match, env)) {
        const char* s = getenv(match[1].str().c_str());
        const std::string var(s == nullptr ? "" : s);
        filename.replace(match[0].first, match[0].second, var);
    }
    
    return filename;
}

std::filesystem::path FilePath::replaceEnvironment(const std::filesystem::path& filename)
{
    return replaceEnvironment(filename.string());
}

void FilePath::init(const char* filename)
{
    m_file = normalizePath(std::string(filename));
}


// ===============================================================
// SubTreeList
// ===============================================================
SubTreeList::SubTreeList(AttributeTree* sub_tree, SubTreeList* next)
    : m_next(next),
    m_sub_tree(sub_tree)
{
}

SubTreeList::~SubTreeList()
{
    if (m_sub_tree)
    {
        // parent auf 0 damit unlink nicht wieder diesen Destruktor aufruft (cycle!)
        m_sub_tree->m_parent = nullptr;
        delete m_sub_tree;
    }
    delete m_next;
}

void SubTreeList::copy(AttributeTree* parent) const
{
    assert(parent != nullptr
        && "SubTreeList::copy() called with nullptr parent! Allocated attribute tree would be lost!");

    const SubTreeList* loop = this;
    while (loop)
    {
        new AttributeTree(*loop->m_sub_tree, parent);
        loop = loop->m_next;
    }
}

void SubTreeList::unlinkParent() const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        if (loop->m_sub_tree)
            loop->m_sub_tree->m_parent = nullptr;

        loop = loop->m_next;
    }
}

void SubTreeList::unlink(AttributeTree* obsolete_tree)
{
	auto loop = this;
    SubTreeList* prev = nullptr;
    while (loop)
    {
        if (loop->m_sub_tree == obsolete_tree)
        {
            if (prev)
            {
                prev->m_next = loop->m_next;
            }
            loop->m_next = nullptr;
            loop->m_sub_tree = nullptr;
            delete loop;
            return;
        }
        prev = loop;
        loop = loop->m_next;
    }
    // nicht gefunden? Dann machen wir gar nichts!
}

AttributeTree* SubTreeList::subTree(const char* description) const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        if (loop->m_sub_tree && loop->m_sub_tree->getDescription()
            && !strcmp(loop->m_sub_tree->getDescription(), description))
        {
            return loop->m_sub_tree;
        }
        loop = loop->m_next;
    }
    return nullptr;
}

int SubTreeList::contains() const
{
    int ret = 0;
    const SubTreeList* loop = this;
    while (loop)
    {
        ret += loop->m_sub_tree->contains();
        loop = loop->m_next;
    }
    return ret;
}

void SubTreeList::printSubTree(std::ostream& out, int change_style_depth, const char* upper_description) const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        loop->m_sub_tree->printSubTree(out, change_style_depth, upper_description);
        loop = loop->m_next;
    }
}

AttributeTree* SubTreeList::search(const char* description, const char* attribute) const
{
    const SubTreeList* loop = this;
    while (loop)
    {
	    if (AttributeTree* search = loop->m_sub_tree->search(description, attribute))
            return search;

        loop = loop->m_next;
    }
    return nullptr;
}

bool SubTreeList::changed() const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        if (loop->m_sub_tree->changed())
            return true;

        loop = loop->m_next;
    }
    return false;
}

AttributeTree* SubTreeList::next(AttributeTree* prev) const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        if (loop->m_sub_tree == prev)
        {
            if (loop->m_next)
            {
                return loop->m_next->m_sub_tree;
            }
        }
        loop = loop->m_next;
    }
    return nullptr;
}

void SubTreeList::unmarkChanges() const
{
    const SubTreeList* loop = this;
    while (loop)
    {
        loop->m_sub_tree->unmarkChanges();
        loop = loop->m_next;
    }
}

SubTreeList* SubTreeList::revertOrder(SubTreeList* new_next)
{
	auto ret = this;
    if (m_sub_tree)
    {
        m_sub_tree->revertOrder();
    }
    if (m_next)
    {
        ret = m_next->revertOrder(this);
    }
    m_next = new_next;
    return ret;
}

void SubTreeList::newSubTreeList(AttributeTree* new_tree, AttributeTree* after)
{
    SubTreeList* loop;
    // loop through all next ones
    for (loop = this; loop->m_next && loop->m_sub_tree != after; loop = loop->m_next)
    {}
    // and insert the new
    loop->m_next = new SubTreeList(new_tree, loop->m_next);
}

// ===============================================================
// AttributeTree
// ===============================================================

AttributeTree::AttributeTree(const char* description, AttributeTree* parent)
    : m_parent(parent),
    m_subtree_list(nullptr)
{
    file_path_str_len = strlen(m_file_path_str);
    file_name_str_len = strlen(m_file_name_str);
    if (description)
    {
        m_this_description = _strdup(description);
    }
    else
    {
        m_this_description = nullptr;
    }
    m_this_attribute = nullptr;
    m_changed = false;

    // Beim Parent in die Liste einfügen
    if (m_parent)
    {
        m_parent->m_subtree_list = new SubTreeList(this, m_parent->m_subtree_list);
    }
}

AttributeTree::AttributeTree(const AttributeTree& tree)
    : m_parent(nullptr),
    m_subtree_list(nullptr)
{
    file_path_str_len = strlen(m_file_path_str);
    file_name_str_len = strlen(m_file_name_str);

    if (tree.m_this_description)
    {
        m_this_description = _strdup(tree.m_this_description);
    }
    else
    {
        m_this_description = nullptr;
    }
    if (tree.m_this_attribute)
    {
        m_this_attribute = _strdup(tree.m_this_attribute);
    }
    else
    {
        m_this_attribute = nullptr;
    }
    if (tree.m_subtree_list)
    {
        tree.m_subtree_list->copy(this);
    }

    m_changed = false;
}

AttributeTree::AttributeTree(const AttributeTree& tree, AttributeTree* parent)
    : m_parent(parent),
    m_subtree_list(nullptr)
{
    file_path_str_len = strlen(m_file_path_str);
    file_name_str_len = strlen(m_file_name_str);

    if (tree.m_this_description)
    {
        m_this_description = _strdup(tree.m_this_description);
    }
    else
    {
        m_this_description = nullptr;
    }
    if (tree.m_this_attribute)
    {
        m_this_attribute = _strdup(tree.m_this_attribute);
    }
    else
    {
        m_this_attribute = nullptr;
    }
    if (tree.m_subtree_list)
    {
        tree.m_subtree_list->copy(this);
    }
    // Beim Parent in die Liste einfügen
    if (m_parent)
    {
        m_parent->m_subtree_list = new SubTreeList(this, m_parent->m_subtree_list);
    }

    m_changed = false;
}

AttributeTree::~AttributeTree()
{
    //  DEBUGMSG(-3, "AttributeTree::~ >> Deleting ...\n");
    if (m_this_description)
    {
        // DEBUGMSG(-3, "\t descr(%p)='%s'\n", this, m_this_description);
        free(m_this_description);
        m_this_description = nullptr;
    }
    if (m_this_attribute)
    {
        //      DEBUGMSG(-3, "\t attr=%s\n", m_this_attribute);
        free(m_this_attribute);
        m_this_attribute = nullptr;
    }
    // subtree wird komplett ausgelöscht
    if (m_subtree_list)
    {
        // DEBUGMSG(-3, "Entering sub (%p)...\n", this);
        delete m_subtree_list;
        m_subtree_list = nullptr;
        // DEBUGMSG(-3, "Leaving sub (%p) ...\n", this);
    }

    unlink();
}

void AttributeTree::unlinkSub()
{
    if (m_subtree_list)
    {
        // die parent-Zeiger der Unterbäume auf 0 setzen
        m_subtree_list->unlinkParent();
        // den Unterbaumzeiger auf 0
        m_subtree_list = nullptr;
    }
}

void AttributeTree::unlink()
{
    if (m_parent)
    {
        SubTreeList* first_entry = m_parent->m_subtree_list;
        if (first_entry->m_sub_tree == this)
        {
            m_parent->m_subtree_list = first_entry->m_next;
        }

        first_entry->unlink(this);
        m_parent->m_changed = true;
    }
    m_parent = nullptr;
}


void AttributeTree::setDescription(const char* description)
{
    free(m_this_description);
    if (description)
    {
        m_this_description = _strdup(description);
    }
    else
    {
        m_this_description = nullptr;
    }
}

void AttributeTree::setAttribute(const char* attribute)
{
    //printf("Change Attribute:%s %s\n",m_this_attribute,attribute);
    if (!m_this_attribute || !attribute || strcmp(attribute, m_this_attribute))
    {
        free(m_this_attribute);
        if (attribute)
        {
            m_this_attribute = _strdup(attribute);
        }
        else
        {
            m_this_attribute = nullptr;
        }
        m_changed = true;
    }
}


AttributeTree* AttributeTree::subTree(const char* description)
{
    AttributeTree* m_sub_tree = getSubTree(description);
    if (m_sub_tree != nullptr)
    {
        // Gibt's schon einen, geben wir diesen zurück
        return m_sub_tree;
    }
    else
    {
        // Ansonsten erzeugen wir einen:
        return new AttributeTree(description, this);
    }
}

AttributeTree* AttributeTree::getSubTree(const char* description) const
{
    AttributeTree* m_sub_tree;
    // Erstmal suchen, obs schon einen gibt:
    if (m_subtree_list)
    {
        m_sub_tree = m_subtree_list->subTree(description);
        if (m_sub_tree)
        {
            return m_sub_tree;
        }
    }
    // Ansonsten geben wir nullptr zurück
    return nullptr;
}


AttributeTree* AttributeTree::setAttribute(const char* param_description, const char* attribute)
{
    if (param_description)
    {
        char* description = _strdup(param_description);
        //printf("a:%s:%s\n",description,attribute);
        char* subdescription;
        split(description, subdescription);
        //printf("b:%s--%p\n",description,subdescription);
        AttributeTree* ret = setAttribute(description, subdescription, attribute);
        free(description);
        //printf("c:%p \n",ret);
        return ret;
    }
    setAttribute(attribute);
    return this;
}

AttributeTree* AttributeTree::setAttribute(const char* description, const char* subdescription,
    const char* attribute)
{
    // printf("%p---%s,%s,%s\n",this,description,subdescription,attribute);
    if (!description || !*description)
    {
        // Keine Description -> Wir sind am Endknoten -> Eintrag machen
        //  printf("set attribute: %s :%s\n",m_this_description,attribute);
        setAttribute(attribute);

        return this;
    }

    //printf("1--%p\n",m_this_description);
    // Ansonsten müssen wir weiter nach unten suchen:
    AttributeTree* subtree = nullptr;
    if (m_subtree_list)
    {
        subtree = m_subtree_list->subTree(description);
    }

    //printf("2--\n");
    if (subtree)
    {
        return subtree->setAttribute(subdescription, attribute);
    }

    // Kein passender Eintrag gefunden -> neuen Sub-Baum erzeugen:
    const auto new_subtree = new AttributeTree(description, this);
    //printf("3--:%p\n",new_subtree);

    return new_subtree->setAttribute(subdescription, attribute);
}

char* AttributeTree::getSpecialAttribute(const char* description, AttributeTree** subtree) const
{
    // search recursive to the root for that attribute
    const AttributeTree* at_path_parent = this;
    AttributeTree* at_path = at_path_parent->m_subtree_list->subTree(description);
    while (at_path_parent && at_path == nullptr)
    {
        at_path = at_path_parent->m_subtree_list->subTree(description);
        at_path_parent = at_path_parent->parentTree();
    }

    // found
    if (at_path && at_path->m_this_attribute)
    {
        //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getSpecialAttribute>> found special attribute %s with %s\n",
        //         m_file_path_str, at_path->m_this_attribute);
        if (subtree)
        {
            (*subtree) = at_path;
        }
        return at_path->m_this_attribute;
    }
    return nullptr;
}

char* AttributeTree::getAttribute(const char* param_description, const char* default_attribute,
    AttributeTree** subtree)
{
    char* ret = nullptr;
    if (param_description)
    {
        char* description = _strdup(param_description);
        if (description)
        {
	        auto at = this;
            // check for 'm_file_path_str' and 'm_file_name_str'
            const size_t len = strlen(description);
            if (len >= file_path_str_len
                && !strncmp(description + (len - file_path_str_len), m_file_path_str, file_path_str_len))
            {
                ret = getSpecialAttribute(m_file_path_str, subtree);
            }
            else if (len >= file_name_str_len
                && !strncmp(description + (len - file_name_str_len), m_file_name_str, file_name_str_len))
            {
                ret = getSpecialAttribute(m_file_name_str, subtree);
            }

            // not found yet ... trying the standard search
            if (!ret)
            {
                char* description_part = description;
                // go into the attribute tree structure
                while (at && description_part)
                {
                    // save the begin of the description
                    const char* next_description = description_part;
                    // searching for further dots
                    description_part = strchr(description_part, '.');
                    if (description_part)
                    {
                        *description_part = 0;
                        description_part++;
                    }
                    at = at->m_subtree_list->subTree(next_description);
                }
                // now we are at the inner attribute tree
                // is there an attribute
                if (at && at->m_this_attribute)
                {
                    //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getAttribute>> found %s\n", at->m_this_attribute);
                    if (subtree)
                    {
                        (*subtree) = at;
                    }
                    ret = at->m_this_attribute;
                }
            }
            free(description);
        }
    }
    // didn't find anything
    if (!ret)
    {
        if (subtree)
        {
            (*subtree) = nullptr;
        }
        //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree::getAttribute>> nothing found. Return default %s\n",
        //         default_attribute ? default_attribute : "(null)");
        ret = const_cast<char*>(default_attribute);
    }

    return ret;
}

char* AttributeTree::getOrSetDefault(const char* description, const char* default_attribute)
{
    char* attribute = getAttribute(description, nullptr);
    if (!attribute)
    {
        setAttribute(description, default_attribute);
        attribute = const_cast<char*>(default_attribute);
    }
    return attribute;
}

char* AttributeTree::newSubNodeDescription(const char* base_description) const
{
    const size_t base_len = strlen(base_description);
    const auto description = static_cast<char*>(malloc(base_len + 6));
    assert(description != nullptr); // Just abort if we are out of memory.
    strcpy_s(description, base_len + 6, base_description);
    int i = 1;
    int j = 0;

    // find the maxima length of number in base_description
    if (base_len > 0)
    {
        while (base_len >= j - 1 &&
            sscanf_s(description + base_len - j - 1, "%i", &i) == 1)
        {
            j++;
        }
        if (j != 0)
        {
            i++;
        }
    }

    sprintf(description + base_len - j, "%i", i);

    while (m_subtree_list->subTree(description) && i < 100000)
    {
        i++;
        sprintf(description + base_len - j, "%i", i);
    }
    return description;
}


AttributeTree* AttributeTree::addNewSubTree()
{
    char* name = newSubNodeDescription();
    AttributeTree* ret = setAttribute(name, nullptr);
    free(name);
    return ret;
}


AttributeTree* AttributeTree::addSubTree(AttributeTree* tree, AttributeTree* after)
{
    if (tree)
    {
        if (m_subtree_list->subTree(tree->m_this_description))
        {
            char* new_description = newSubNodeDescription(tree->m_this_description);
            free(tree->m_this_description);
            tree->m_this_description = new_description;
        }

        if (after == nullptr)
        {
            m_subtree_list = new SubTreeList(tree, m_subtree_list);
        }
        else
        {
            m_subtree_list->newSubTreeList(tree, after);
        }

        tree->m_parent = this;
        return tree;
    }
    else
    {
        return nullptr;
    }
}

void AttributeTree::printSubTree(std::ostream& out, int change_style_depth, const char* upper_description)
{
    // virtual attributes are not stored !
    if (m_this_description && (!strcmp(m_this_description, m_file_path_str)
        || !strcmp(m_this_description, m_file_name_str)))
    {
        return;
    }

    char* the_upper_description = _strdup(upper_description ? upper_description : "");
    char* t_description = _strdup(m_this_description ? m_this_description : "");
    assert(the_upper_description != nullptr);
    assert(t_description != nullptr);

    // is this the comment attribute tree ?
    if (isMultilineComment())
    {
        out << the_upper_description << comment_str << '{' << std::endl;
        out << m_this_attribute << std::endl;
        out << the_upper_description << '}' << comment_str << std::endl;

        free(the_upper_description);
        free(t_description);

        return;
    }

    const int contents = contains();
    if (contents >= change_style_depth || hasMultilineComment())
    {
        out << the_upper_description << t_description << '{' << std::endl;
        if (m_this_attribute && strcmp(m_this_attribute, ""))
        {
            out << the_upper_description << ':' << m_this_attribute << std::endl;
        }

        if (m_subtree_list)
        {
	        const auto tab = static_cast<char*>(malloc(strlen(the_upper_description) + 2));
            assert(tab != nullptr); // Just abort if we are out of memory.
            strcat(strcpy(tab, the_upper_description), " ");
            m_subtree_list->printSubTree(out, change_style_depth, tab);
            free(tab);
        }

        out << the_upper_description << '}' << t_description << std::endl;
    }
    else
    {
        const size_t tud_len = strlen(the_upper_description);
        const size_t len = strlen(t_description) + tud_len + 1;
        const auto description = static_cast<char*>(malloc(len + 1));
        assert(description != nullptr); // Just abort if we are out of memory.
        memset(description, 0, len + 1);

        if ((tud_len > 0) && (the_upper_description[tud_len - 1] == ' '))
        {
            strcat(strcpy(description, the_upper_description), t_description);
        }
        else
        {
            strcat(strcat(strcpy(description, the_upper_description), "."), t_description);
        }

        if (m_this_attribute)
        {
            out << description << ':' << m_this_attribute << std::endl;
        }

        if (m_subtree_list)
        {
            m_subtree_list->printSubTree(out, change_style_depth, description);
        }
        free(description);
    }

    free(the_upper_description);
    free(t_description);
}

int AttributeTree::save(const char* filename, int change_style_depth, bool unmark_changes)
{
    /*
    if (!m_this_description)
      return eEMPTY_TREE;
    */
    std::ofstream out(filename);
    if (!out)
    {
        return eFILE_SAVE_ERROR;
    }
    printSubTree(out, change_style_depth, "");

    if (unmark_changes)
    {
        unmarkChanges();
    }

    return eOK;
}

int AttributeTree::load(const char* filename, bool unmark_changes, bool process_include,
    bool load_comments, bool preserve_order)
{
    if (filename == nullptr || strcmp(filename, "") == 0)
    {
        printf("tAttributeTree >> Trying to load an empty configuration file.\n");
        return eFILE_LOAD_ERROR;
    }

    const icl_core::config::FilePath at_file(filename);
    //LOCAL_PRINTF("AttributeTree >> Loading %s\n", at_file.AbsoluteName().c_str());
    if (this == root() && !getAttribute(m_file_path_str))
    {
        //LOCAL_PRINTF("AttributeTree >> Setting Virtual Attributes Path(%s) Name(%s)\n",
        //             at_file.Path().c_str(), at_file.Name().c_str());
        setAttribute(m_file_path_str, at_file.path().string().c_str());
        setAttribute(m_file_name_str, at_file.name().string().c_str());
    }

    int error;
    std::ifstream in(at_file.absoluteName().c_str());
    if (!in)
    {
        printf("tAttributeTree >> Could not open file '%s'\n", at_file.absoluteName().string().c_str());
        return eFILE_LOAD_ERROR;
    }

    error = get(in, process_include, load_comments, &at_file);
    if (error >= 0)
    {
        printf("Error in line %i while reading AttributeTree %s\n", error,
            at_file.absoluteName().string().c_str());
        return eFILE_LOAD_ERROR;
    }


    if (unmark_changes)
    {
        unmarkChanges();
    }
    if (preserve_order)
    {
        revertOrder();
    }
    //DEBUGMSG(DD_AT, DL_DEBUG, "AttributeTree >> Loading successful\n");

    return eOK;
}

int AttributeTree::get(std::istream& in, bool process_include, bool load_comments,
                       const FilePath* file_path)
{
    // save stack memory on recursive calls!
    // without static in the insmod call for the RTL-module we crash !
    std::string line;
    
    auto at = this;
    int lineno = 1;

    readNextLineInBuffer(in);

    do
    {
        //LOCAL_PRINTF("get next line %i\n",lineno);
        lineno++;
        line = buffer;
        boost::algorithm::trim_left(line);
        
        //LOCAL_PRINTF("%s\n",line);
        if (line[0] != '#')
        {
	        auto pot_attribute_pos = line.find(':');

            //attribute = strchr(line, ':');
            if (pot_attribute_pos != std::string::npos)
            {
                line[pot_attribute_pos] = 0;
                if (!line[0])
                {
                    //LOCAL_PRINTF("AttributeTree::get >> found ':%s'\n", attribute+1);
                    at->setAttribute(line.substr(pot_attribute_pos + 1).c_str());
                }
                else
                {
                    if (line == include_str)
                    {
                        if (process_include)
                        {
                            auto include_filename = std::filesystem::path(line.substr(include_str.size()));
                            include_filename = FilePath::replaceEnvironment(include_filename);
                            if (FilePath::isRelativePath(include_filename))
                            {
                                auto absolute_include_filename(file_path ? file_path->path() : getFilePath());
                                absolute_include_filename += include_filename;
                                include_filename = FilePath::normalizePath(absolute_include_filename);
                            }
                            if (at->load(include_filename.string().c_str(), false, process_include, load_comments) != eOK)
                            {
                                std::cout << "error loading include file " << include_filename << std::endl;
                            }
                        }
                        else
                        {
                            // falls nicht includen: als "normalen" Eintrag speichern
                            (new AttributeTree(include_str.c_str(), at))->setAttribute(line.substr(include_str.size()).c_str());
                        }
                    }
                    else if (line == comment_str || load_comments)
                    {
                        //LOCAL_PRINTF("AttributeTree::get >> found '%s:%s'\n", line, attribute+1);
                        at->setAttribute(line.c_str(), line.substr(pot_attribute_pos + 1).c_str());
                    }
                }
            }
            else
            {
                pot_attribute_pos = line.find('{');

                //attribute = strchr(line, '{');
                if (pot_attribute_pos != std::string::npos)
                {
                    line[pot_attribute_pos] = 0;
                    //*attribute = 0;
                    //LOCAL_PRINTF("AttributeTree::get >> found '%s{'\n",  line);
                    // multiline comments
                    if (line == comment_str)
                    {
                        AttributeTree* at_c = nullptr;
                        bool comment_end;
                        if (load_comments)
                        {
                            at_c = new AttributeTree(comment_str.c_str(), at);
                        }
                        do
                        {
                            lineno++;
                            readNextLineInBuffer(in);
                            line = buffer;

                            boost::algorithm::trim_right(line);
                            comment_end = line.ends_with(comment_end_str);

                            if (load_comments && !comment_end)
                            {
                                at_c->appendAttribute(line.c_str(), "\n");
                            }
                        } while (!comment_end);
                    }
                    else
                    {
                        at = at->setAttribute(line.c_str(), nullptr);
                    }
                }
                else
                {
                    pot_attribute_pos = line.find('}');
                    if (pot_attribute_pos != std::string::npos)
                    {
                        if (at == this)
                        {
                            //LOCAL_PRINTF("AttributeTree::get >> found last '}'\n");
                            return -1;
                        }
                        else
                        {
                            //LOCAL_PRINTF("AttributeTree::get >> found '}'\n");
                            if (!at->parentTree())
                            {
                                return lineno;
                            }
                            at = at->parentTree();
                        }
                    }
                    else
                    {
                        if (!in.eof() && line[0])
                        {
                            //LOCAL_PRINTF("AttributeTree::get >> found '%s' and could not interpret\n", line);
                            return lineno;
                        }
                    }
                }
            }
        }
        readNextLineInBuffer(in);
    } while (!in.eof());
    return -1;
}

void AttributeTree::split(char*& description, char*& subdescription) const
{
    subdescription = strchr(description, '.');
    if (subdescription)
    {
        *subdescription = 0;
        subdescription++;
    }
}

AttributeTree* AttributeTree::search(const char* description, const char* attribute)
{
    if (description)
    {
        if ((m_this_description && (!strcmp(description, m_this_description))) &&
            (attribute == nullptr || (m_this_attribute && (!strcmp(attribute, m_this_attribute)))))
        {
            return this;
        }
        if (m_subtree_list)
            return m_subtree_list->search(description, attribute);
    }
    return nullptr;
}

bool AttributeTree::isAttribute(const char* description, const char* attribute)
{
    const char* content = getAttribute(description, nullptr);

    if (attribute)
    {
        if (content)
        {
            return !strcmp(content, attribute);
        }
        return false;
    }
    return content != nullptr;
}

bool AttributeTree::changed() const
{
    if (m_changed)
    {
        return true;
    }
    if (m_subtree_list)
    {
        return m_subtree_list->changed();
    }
    return false;
}

void AttributeTree::unmarkChanges()
{
    m_changed = false;
    if (m_subtree_list)
    {
        m_subtree_list->unmarkChanges();
    }
}

int AttributeTree::contains() const
{
    int ret = 0;
    if (m_this_attribute)
    {
        ret++;
    }
    if (m_subtree_list)
    {
        ret += m_subtree_list->contains();
    }
    return ret;
}

void AttributeTree::appendString(char*& dest, const char* src, const char* additional_separator)
{
    if (!src)
    {
        return;
    }
    if (!additional_separator)
    {
        additional_separator = "";
    }
    if (dest)
    {
        const size_t old_len = strlen(dest);
        const size_t additional_len = strlen(additional_separator);
        const size_t whole_len = old_len + additional_len + strlen(src);
        const auto new_attr = static_cast<char*>(malloc(whole_len + 1));
        assert(new_attr != nullptr); // Just abort if out of memory!
        strcpy(new_attr, dest);
        strcpy(new_attr + old_len, additional_separator);
        strcpy(new_attr + old_len + additional_len, src);
        free(dest);
        dest = new_attr;
    }
    else
    {
        dest = _strdup(src);
    }
    m_changed = true;
}

AttributeTree* AttributeTree::commentAttributeTree() const
{
    AttributeTree* loop = firstSubTree();
    while (loop)
    {
        if (loop->isComment() && loop->attribute())
        {
            return loop;
        }
        loop = nextSubTree(loop);
    }
    return nullptr;
}

bool AttributeTree::isComment() const
{
    return m_this_description && !strcmp(m_this_description, comment_str.c_str());
}

bool AttributeTree::isMultilineComment() const
{
    return isComment() && (strchr(attribute(), '\n') != nullptr);
}

const char* AttributeTree::comment() const
{
	if (const AttributeTree* sub_comment = commentAttributeTree())
        return sub_comment->attribute();

    return "";
}

bool AttributeTree::hasMultilineComment() const
{
    return strchr(comment(), '\n') != nullptr;
}

void AttributeTree::setComment(const char* comment)
{
    setAttribute(comment_str.c_str(), comment);
}

}
}
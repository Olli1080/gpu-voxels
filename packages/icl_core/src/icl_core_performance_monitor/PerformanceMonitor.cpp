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
 * \author  Florian Drews
 * \date    2013-11-16
 *
 */
//----------------------------------------------------------------------/*

#include "icl_core_performance_monitor/PerformanceMonitor.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace icl_core {
    namespace perf_mon {

        std::string makeName(const std::string& prefix, const std::string& name)
        {
            return prefix + "::" + name;
        }

        PerformanceMonitor* PerformanceMonitor::m_instance = nullptr;

        PerformanceMonitor::PerformanceMonitor()
        {
            m_enabled = true;
            m_print_stop = true;
            m_all_enabled = false;
        }

        PerformanceMonitor::~PerformanceMonitor() = default;

        PerformanceMonitor* PerformanceMonitor::getInstance()
        {
            if (m_instance == nullptr)
                m_instance = new PerformanceMonitor();

            return m_instance;
        }

        void PerformanceMonitor::initialize(const uint32_t num_names, const uint32_t num_events)
        {
            PerformanceMonitor* monitor = getInstance();
            monitor->m_data.clear();
            monitor->m_data_nontime.clear();
            monitor->m_buffer.resize(num_names);
            for (uint32_t i = 0; i < num_names; ++i)
                monitor->m_buffer[i].reserve(num_events);

            icl_core::perf_mon::PerformanceMonitor::enablePrefix("");
        }


        bool PerformanceMonitor::isEnabled(const std::string& prefix) const
        {
            return (m_enabled && m_enabled_prefix.contains(prefix)) || m_all_enabled;
        }

        void PerformanceMonitor::start(const std::string& timer_name)
        {
            if (!getInstance()->m_enabled)
                return;

            const TimeStamp t = TimeStamp::now();
            getInstance()->m_timer[timer_name] = t;
        }

        double PerformanceMonitor::measurement(const std::string& timer_name, const std::string& description,
                                               const std::string& prefix,
                                               icl_core::logging::LogLevel level)
        {
            PerformanceMonitor* monitor = getInstance();
            if (monitor->isEnabled(prefix))
            {
                const TimeStamp end = TimeStamp::now();
                const auto d(end - monitor->m_timer[timer_name]);
                const auto double_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d);
                monitor->addEvent(prefix, description, double_ms.count());

                if (getInstance()->m_print_stop)
                {
                    std::stringstream ss;
                    ss << makeName(prefix, description) << ": " << double_ms << " ms";
                    monitor->print(ss.str(), level);
                }
                return double_ms.count();
            }
            return 0;
        }

        double PerformanceMonitor::startStop(const std::string& timer_name, const std::string& description,
                                             const std::string& prefix,
                                             logging::LogLevel level, const bool silent)
        {
            /*
             * If timer_name exists:
             *   stop timer
             *   make new start time equal to stop time
             * else
             *   start timer
             */

            if (!getInstance()->isEnabled(prefix))
                return 0;

            PerformanceMonitor* monitor = getInstance();
            const TimeStamp start = monitor->m_timer[timer_name];
            if (start != TimeStamp())
            {
                const TimeStamp end = TimeStamp::now();
                const auto d(end - start);
                const auto double_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(d);
                monitor->addEvent(prefix, description, double_ms.count());
                monitor->m_timer[timer_name] = end;
                if (!silent && getInstance()->m_print_stop)
                {
                    std::stringstream ss;
                    ss << makeName(prefix, description) << ": " << double_ms << " ms";
                    monitor->print(ss.str(), level);
                }
                return double_ms.count();
            }
            else
            {
                PerformanceMonitor::start(timer_name);
                return 0;
            }
        }

        void PerformanceMonitor::addStaticData(const std::string& name, double data, const std::string& prefix)
        {
            if (getInstance()->isEnabled(prefix))
            {
                const std::string tmp = makeName(prefix, name);
                getInstance()->m_static_data[tmp] = data;
            }
        }

        void PerformanceMonitor::addData(const std::string& name, double data, const std::string& prefix)
        {
            if (getInstance()->isEnabled(prefix))
                getInstance()->addEvent(prefix, name, data);
        }

        void PerformanceMonitor::addNonTimeData(const std::string& name, double data, const std::string& prefix)
        {
            if (getInstance()->isEnabled(prefix))
                getInstance()->addNonTimeEvent(prefix, name, data);
        }

        void PerformanceMonitor::addEvent(const std::string& prefix, const std::string& name, double data)
        {
            if (!getInstance()->isEnabled(prefix))
                return;

            const std::string tmp = makeName(prefix, name);
            if (!m_data.contains(tmp))
            {
                m_data[tmp] = std::vector<double>();
                if (!m_buffer.empty())
                {
                    m_data[tmp].swap(m_buffer.back());
                    m_buffer.pop_back();
                }
            }
            m_data[tmp].push_back(data);
        }

        void PerformanceMonitor::addNonTimeEvent(const std::string& prefix, const std::string& name, double data)
        {
            if (getInstance()->isEnabled(prefix))
            {
                const std::string tmp = makeName(prefix, name);
                if (!m_data_nontime.contains(tmp))
                {
                    m_data_nontime[tmp] = std::vector<double>();
                    if (!m_buffer.empty())
                    {
                        m_data_nontime[tmp].swap(m_buffer.back());
                        m_buffer.pop_back();
                    }
                }
                m_data_nontime[tmp].push_back(data);
            }
        }


        void PerformanceMonitor::print(const std::string& message, logging::LogLevel level)
        {
            switch (level)
            {
            case ::icl_core::logging::eLL_DEBUG:
            {
                LOGGING_DEBUG(Performance, message << endl);
                break;
            }
            case ::icl_core::logging::eLL_INFO:
            {
                LOGGING_INFO(Performance, message << endl);
                break;
            }
            case ::icl_core::logging::eLL_TRACE:
            {
                LOGGING_TRACE(Performance, message << endl);
                break;
            }
            default:
            {
                LOGGING_INFO(Performance, message << endl);
                break;
            }
            }
        }

        void PerformanceMonitor::createStatisticSummary(std::stringstream& ss, const std::string& prefix, const std::string
                                                        & name)
        {
            const std::string tmp = makeName(prefix, name);
            double median, min, max;
            getMedian(tmp, median, min, max);

            ss << "Summary for " << tmp << "\n" <<
                "Called " << m_data[tmp].size() << " times\n" <<
                name << "_avg: " << getAverage(tmp) << " ms\n" <<
                name << "_median: " << median << " ms\n" <<
                name << "_min: " << min << " ms\n" <<
                name << "_max: " << max << " ms\n" <<
                "\n";
        }

        void PerformanceMonitor::createStatisticSummaryNonTime(std::stringstream& ss, const std::string& prefix,
                                                               const std::string& name)
        {
            const std::string tmp = makeName(prefix, name);
            double median, min, max;
            getMedianNonTime(tmp, median, min, max);

            ss << "Summary for " << tmp << "\n" <<
                "num entries: " << m_data_nontime[tmp].size() << "\n" <<
                name << "_avg: " << getAverageNonTime(tmp) << "\n" <<
                name << "_median: " << median << "\n" <<
                name << "_min: " << min << "\n" <<
                name << "_max: " << max << "\n" <<
                "\n";
        }

        std::string PerformanceMonitor::printSummary(const std::string& prefix, const std::string& name,
                                                          icl_core::logging::LogLevel level)
        {
            PerformanceMonitor* monitor = getInstance();

            std::stringstream ss;
            monitor->createStatisticSummary(ss, prefix, name);
            monitor->print(ss.str(), level);
            return ss.str();
        }

        void PerformanceMonitor::enablePrefix(const std::string& prefix)
        {
            PerformanceMonitor* monitor = getInstance();

            if (monitor->m_enabled_prefix.contains(prefix))
                return;

            monitor->m_enabled_prefix[prefix] = true;
        }

        void PerformanceMonitor::enableAll(const bool& enabled)
        {
            getInstance()->m_all_enabled = enabled;
        }

        void PerformanceMonitor::disablePrefix(const std::string& prefix)
        {
            PerformanceMonitor* monitor = getInstance();

            if (monitor->m_enabled_prefix.contains(prefix))
                monitor->m_enabled_prefix.erase(prefix);
        }

        std::string PerformanceMonitor::printSummaryAll(icl_core::logging::LogLevel level)
        {
            PerformanceMonitor* monitor = getInstance();

            std::stringstream ss;
            for (auto it = monitor->m_enabled_prefix.begin();
                it != monitor->m_enabled_prefix.end(); ++it)
            {
                ss << printSummaryFromPrefix(it->first, level);
            }
            return ss.str();
        }

        std::string PerformanceMonitor::printSummaryFromPrefix(const std::string& prefix, icl_core::logging::LogLevel level)
        {
            PerformanceMonitor* monitor = getInstance();
            bool first = true;
            std::stringstream ss;
            ss << "\n########## Begin of Summary for prefix " << prefix << " ##########\n";
            for (auto it = monitor->m_static_data.begin(); it != monitor->m_static_data.end(); ++it)
            {
                size_t prefix_end = it->first.find("::");
                std::string prefix_tmp = it->first.substr(0, prefix_end);

                if (prefix == prefix_tmp)
                {
                    if (first)
                    {
                        ss << "#### Static data: ####\n";
                        first = false;
                    }
                    ss << it->first.substr(prefix_end + 2) << ": " << it->second << "\n";
                }
            }

            first = true;
            for (auto it = monitor->m_data.begin(); it != monitor->m_data.end(); ++it)
            {
                size_t prefix_end = it->first.find("::");
                std::string prefix_tmp = it->first.substr(0, prefix_end);

                if (prefix == prefix_tmp)
                {
                    if (first)
                    {
                        ss << "#### Time data: ####\n";
                        first = false;
                    }
                    std::string name = it->first.substr(prefix_end + 2);
                    monitor->createStatisticSummary(ss, prefix, name);
                }
            }

            first = true;
            for (auto it = monitor->m_data_nontime.begin(); it != monitor->m_data_nontime.end(); ++it)
            {
                size_t prefix_end = it->first.find("::");
                std::string prefix_tmp = it->first.substr(0, prefix_end);

                if (prefix == prefix_tmp)
                {
                    if (first)
                    {
                        ss << "#### Non-time data: ####\n";
                        first = false;
                    }
                    std::string name = it->first.substr(prefix_end + 2);
                    monitor->createStatisticSummaryNonTime(ss, prefix, name);
                }
            }
            monitor->print(ss.str(), level);
            return ss.str();
        }

        double PerformanceMonitor::getAverage(const std::string& name)
        {
            double avg = 0;
            const std::vector<double>* tmp = &m_data[name];
            const int n = static_cast<int>(m_data[name].size());
            for (int i = 0; i < n; ++i)
                avg = avg + tmp->at(i);
            avg = avg / n;
            return avg;
        }

        double PerformanceMonitor::getAverageNonTime(const std::string& name)
        {
            double avg = 0;
            const std::vector<double>* tmp = &m_data_nontime[name];
            const int n = static_cast<int>(m_data_nontime[name].size());
            for (int i = 0; i < n; ++i)
                avg = avg + tmp->at(i);
            avg = avg / n;
            return avg;
        }

        void PerformanceMonitor::getMedian(const std::string& name, double& median, double& min, double& max)
        {
	        std::vector<double> tmp = m_data[name];
	        std::ranges::sort(tmp);
            median = tmp[tmp.size() / 2];
            min = tmp.front();
            max = tmp.back();
        }

        void PerformanceMonitor::getMedianNonTime(const std::string& name, double& median, double& min, double& max)
        {
	        std::vector<double> tmp = m_data_nontime[name];
	        std::ranges::sort(tmp);
            if (!tmp.empty())
                median = tmp[tmp.size() / 2];
            min = tmp.front();
            max = tmp.back();
        }

        std::vector<double> PerformanceMonitor::getData(const std::string& name, const std::string& prefix)
        {
            PerformanceMonitor* monitor = getInstance();
            return monitor->m_data[makeName(prefix, name)];
        }

        std::vector<double> PerformanceMonitor::getNonTimeData(const std::string& name, const std::string&
                                                               prefix)
        {
            PerformanceMonitor* monitor = getInstance();
            return monitor->m_data_nontime[makeName(prefix, name)];
        }

    } // namespace timer
} // namespace icl_core
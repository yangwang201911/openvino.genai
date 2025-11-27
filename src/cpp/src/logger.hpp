// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "openvino/openvino.hpp"

namespace ov::genai {
ov::log::Level get_openvino_env_log_level();
ov::log::Level get_cur_log_level();

class Logger {
public:
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }
    ~Logger() = default;
    void do_log(ov::log::Level level, const char* file, int line, const std::string& msg);

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 5, 6)))
#endif
    void log_format(ov::log::Level level, const char* file, int line, const char* format, ...);

    void set_log_level(ov::log::Level level);

    bool should_log(ov::log::Level level) const;

private:
    Logger();
    std::atomic<ov::log::Level> log_level{ov::log::Level::NO};
    std::mutex log_mutex;
    void write_message(ov::log::Level level, const char* file, int line, const std::string& msg);
    void log_format_impl(ov::log::Level level, const char* file, int line, const char* format, va_list args);
    std::string format_from_variadic(const char* format, va_list args) const;
    std::string_view get_filename(std::string_view file_path) const;
    std::ostream& format_prefix(std::ostream& out, ov::log::Level level, const char* file, int line) const;
};
namespace detail {

// Single string argument
inline void log_message(ov::log::Level level, const char* file, int line, const std::string& msg) {
    auto& logger = Logger::get_instance();
    if (!logger.should_log(level)) {
        return;
    }
    logger.do_log(level, file, line, msg);
}

// Single const char* argument
inline void log_message(ov::log::Level level, const char* file, int line, const char* msg) {
    auto& logger = Logger::get_instance();
    if (!logger.should_log(level)) {
        return;
    }
    logger.do_log(level, file, line, msg ? std::string(msg) : std::string());
}

// Helper to build concatenated message from multiple arguments
inline void build_message(std::ostringstream&) {}

template <typename T, typename... Args>
inline void build_message(std::ostringstream& oss, T&& first, Args&&... rest) {
    oss << std::forward<T>(first);
    build_message(oss, std::forward<Args>(rest)...);
}

// Type trait to check if a string looks like a format string (contains %)
inline bool is_format_string(const char* str) {
    return str && std::strchr(str, '%') != nullptr;
}

// Printf-style format with arguments (const char* format followed by at least one argument)
// This overload has higher priority due to more specific type constraints
template <typename Arg1, typename... Args>
inline auto log_message(ov::log::Level level,
                        const char* file,
                        int line,
                        const char* format,
                        Arg1&& arg1,
                        Args&&... args)
    -> std::enable_if_t<(std::is_arithmetic_v<std::decay_t<Arg1>> || std::is_pointer_v<std::decay_t<Arg1>> ||
                         std::is_same_v<std::decay_t<Arg1>, const char*>),
                        void> {
    auto& logger = Logger::get_instance();
    if (!logger.should_log(level)) {
        return;
    }
    logger.log_format(level, file, line, format, std::forward<Arg1>(arg1), std::forward<Args>(args)...);
}

// String concatenation for string literal + other arguments (lower priority)
template <size_t N, typename U, typename... Args>
inline auto log_message(ov::log::Level level,
                        const char* file,
                        int line,
                        const char (&str_literal)[N],
                        U&& second,
                        Args&&... rest)
    -> std::enable_if_t<!std::is_arithmetic_v<std::decay_t<U>> && !std::is_pointer_v<std::decay_t<U>> &&
                            !std::is_same_v<std::decay_t<U>, const char*>,
                        void> {
    auto& logger = Logger::get_instance();
    if (!logger.should_log(level)) {
        return;
    }
    std::ostringstream oss;
    build_message(oss, str_literal, std::forward<U>(second), std::forward<Args>(rest)...);
    logger.do_log(level, file, line, oss.str());
}

// String concatenation for multiple arguments where first arg is NOT const char* or string literal
template <typename T, typename U, typename... Args>
inline auto log_message(ov::log::Level level, const char* file, int line, T&& first, U&& second, Args&&... rest)
    -> std::enable_if_t<!std::is_same_v<std::decay_t<T>, const char*> && !std::is_same_v<std::decay_t<T>, char*> &&
                            !std::is_array_v<std::remove_reference_t<T>>,
                        void> {
    auto& logger = Logger::get_instance();
    if (!logger.should_log(level)) {
        return;
    }
    std::ostringstream oss;
    build_message(oss, std::forward<T>(first), std::forward<U>(second), std::forward<Args>(rest)...);
    logger.do_log(level, file, line, oss.str());
}

}  // namespace detail

#define GENAI_DEBUG(...) ::ov::genai::detail::log_message(ov::log::Level::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define GENAI_INFO(...)  ::ov::genai::detail::log_message(ov::log::Level::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define GENAI_WARN(...)  ::ov::genai::detail::log_message(ov::log::Level::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define GENAI_ERR(...)   ::ov::genai::detail::log_message(ov::log::Level::ERR, __FILE__, __LINE__, __VA_ARGS__)

}  // namespace ov::genai

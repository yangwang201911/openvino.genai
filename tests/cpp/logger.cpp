// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logger.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace {

void expect_contains(const std::string& haystack, const std::string& needle) {
    EXPECT_NE(haystack.find(needle), std::string::npos);
}

class LoggerTests : public ::testing::Test {
protected:
    void SetUp() override {
        ov::genai::Logger::get_instance().set_log_level(ov::log::Level::DEBUG);
    }

    void TearDown() override {
        ov::genai::Logger::get_instance().set_log_level(ov::log::Level::NO);
    }
};

}  // namespace

TEST_F(LoggerTests, SupportsPrintfStyleFormatting) {
    testing::internal::CaptureStdout();
    GENAI_INFO("The value of %s is %d", "alpha", 42);
    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] ");
    expect_contains(output, "The value of alpha is 42");
    ASSERT_FALSE(output.empty());
    EXPECT_EQ('\n', output.back());
}

TEST_F(LoggerTests, KeepsSingleTrailingNewline) {
    testing::internal::CaptureStdout();
    GENAI_INFO("Message with newline\n");
    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] ");
    EXPECT_EQ(1, std::count(output.begin(), output.end(), '\n'));
}

TEST_F(LoggerTests, ValidFormatDoesNotThrow) {
    EXPECT_NO_THROW(GENAI_INFO("Hello, %s", "world"));
    EXPECT_NO_THROW(GENAI_INFO("Hello, %s\n", "world"));
    EXPECT_NO_THROW(GENAI_INFO("%d + %d = %d", 1, 2, 3));
    EXPECT_NO_THROW(GENAI_INFO("Pi is approximately %.2f", 3.14159));
    EXPECT_NO_THROW(GENAI_INFO("Hex: 0x%X", 255));
    EXPECT_NO_THROW(GENAI_INFO("Pointer %p", static_cast<const void*>(this)));

    int value1 = 42;
    float value2 = 78.6f;
    std::string text = "result";

    EXPECT_NO_THROW(GENAI_INFO(text, ": ", value1, " ms"));
    EXPECT_NO_THROW(GENAI_INFO("Info Processing ", text, " with float ", value2));
    EXPECT_NO_THROW(GENAI_WARN("Warning Processing ", text, " with float ", value2));
    EXPECT_NO_THROW(GENAI_ERR("Error Processing ", text, " with float ", value2));
    EXPECT_NO_THROW(GENAI_DEBUG(std::string("Mixed: "), value1, " and ", value2, " end"));
}

TEST_F(LoggerTests, NoOutputWhenLevelIsNo) {
    ov::genai::Logger::get_instance().set_log_level(ov::log::Level::NO);
    testing::internal::CaptureStdout();
    GENAI_INFO("Should not appear");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.empty());
}

TEST_F(LoggerTests, RespectsLogLevelFiltering) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();

    ov::genai::Logger::get_instance().set_log_level(ov::log::Level::WARNING);
    GENAI_DEBUG("debug message");
    GENAI_INFO("info message");
    GENAI_WARN("warn message");
    GENAI_ERR("error message");

    std::string output = testing::internal::GetCapturedStdout();
    std::string error_output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(output.find("debug message"), std::string::npos);
    EXPECT_EQ(output.find("info message"), std::string::npos);
    expect_contains(output, "[WARNING] warn message");
    EXPECT_EQ(error_output.find("debug message"), std::string::npos);
    EXPECT_EQ(error_output.find("info message"), std::string::npos);
    EXPECT_TRUE(error_output.find("[WARNING]") == std::string::npos);
    expect_contains(error_output, "[ERROR] error message");
}

TEST_F(LoggerTests, EmitsAllLogLevelsWithoutErrors) {
    ov::genai::Logger::get_instance().set_log_level(ov::log::Level::DEBUG);
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();

    GENAI_DEBUG("debug level message");
    GENAI_INFO("info level message");
    GENAI_WARN("warning level message");
    GENAI_ERR("error level message");

    std::string std_output = testing::internal::GetCapturedStdout();
    std::string err_output = testing::internal::GetCapturedStderr();

    // DEBUG output should contain timestamp and file:line
    expect_contains(std_output, "[DEBUG]");
    expect_contains(std_output, ":");
    expect_contains(std_output, "debug level message");
    // INFO and WARNING should not contain timestamp
    expect_contains(std_output, "[INFO] info level message");
    expect_contains(std_output, "[WARNING] warning level message");

    EXPECT_TRUE(err_output.find("debug level message") == std::string::npos);
    EXPECT_TRUE(err_output.find("info level message") == std::string::npos);
    EXPECT_TRUE(err_output.find("warning level message") == std::string::npos);
    expect_contains(err_output, "[ERROR] error level message");
}

TEST_F(LoggerTests, SupportsVariadicStringConcatenation) {
    testing::internal::CaptureStdout();

    int value1 = 42;
    int value2 = 100;
    std::string text = "result";

    GENAI_INFO(text, ": ", value1, " ms");
    GENAI_INFO("Processing ", text, " with count ", value2);
    GENAI_DEBUG(std::string("Mixed: "), value1, " and ", value2, " end");

    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] result: 42 ms");
    expect_contains(output, "[INFO] Processing result with count 100");
    expect_contains(output, "Mixed: 42 and 100 end");
}

TEST_F(LoggerTests, HandlesEmptyStringsInConcatenation) {
    testing::internal::CaptureStdout();

    std::string empty = "";
    GENAI_INFO(std::string("Start"), empty, " middle ", 123, empty);

    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "[INFO] Start middle 123");
}

TEST_F(LoggerTests, ConcatenatesMultipleTypes) {
    testing::internal::CaptureStdout();

    int int_val = 10;
    long long_val = 20L;
    float float_val = 3.14f;
    double double_val = 2.718;
    std::string str = "string";

    GENAI_INFO(std::string("Int: "),
               int_val,
               ", Long: ",
               long_val,
               ", Float: ",
               float_val,
               ", Double: ",
               double_val,
               ", Str: ",
               str);

    std::string output = testing::internal::GetCapturedStdout();

    expect_contains(output, "Int: 10");
    expect_contains(output, "Long: 20");
    expect_contains(output, "Float: 3.14");
    expect_contains(output, "Double: 2.718");
    expect_contains(output, "Str: string");
}


#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <string>
#include <iostream>


// 日志宏定义
#define LOG_TRACE(...)    spdlog::trace(__VA_ARGS__)
#define LOG_DEBUG(...)    spdlog::debug(__VA_ARGS__)
#define LOG_INFO(...)     spdlog::info(__VA_ARGS__)
#define LOG_WARN(...)     spdlog::warn(__VA_ARGS__)
#define LOG_ERROR(...)    spdlog::error(__VA_ARGS__)
#define LOG_CRITICAL(...) spdlog::critical(__VA_ARGS__)

// 日志类
class Logger {
public:
    static void init(const std::string& log_file = "logs/vision_guide.log") {
        try {
            // 控制台输出
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%H:%M:%S] %^[%l]%$ %v");

            // 文件输出（自动轮转，最大5MB，保留3个文件）
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_file, 1024 * 1024 * 5, 3
            );
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

            // 组合日志器
            auto logger = std::make_shared<spdlog::logger>("vision_guide",
                spdlog::sinks_init_list{ console_sink, file_sink });

#ifdef _DEBUG
            logger->set_level(spdlog::level::debug);
#else
            logger->set_level(spdlog::level::info);
#endif

            logger->flush_on(spdlog::level::info);
            spdlog::set_default_logger(logger);

            LOG_INFO("日志系统初始化完成");

        }
        catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "日志初始化失败: " << ex.what() << std::endl;
        }
    }
};
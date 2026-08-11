#pragma once

#include <concepts>
#include <cstddef>
#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace loki::error_check {
class DetailedException : public std::runtime_error {
public:
    explicit DetailedException(
        std::string_view user_msg,
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(compose_message(user_msg, loc)) {}

private:
    static std::string compose_message(std::string_view msg,
                                       const std::source_location& loc) {
        return std::format("{}:{}:{}: error: {}\n  in function '{}'",
                           loc.file_name(), loc.line(), loc.column(), msg,
                           loc.function_name());
    }
};

// General check with custom comparator and message
template <typename T, typename U, typename Comparator>
inline void check_with_comparator(
    const T& a,
    const U& b,
    Comparator cmp,
    std::string_view op_str,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    if (!cmp(a, b)) {
        std::string composed =
            msg.empty() ? std::format("Check failed: {} {} {}", a, op_str, b)
                        : std::format("{} ({} {} {})", msg, a, op_str, b);
        throw DetailedException(composed, loc);
    }
}

// Specific relational checks
// Check if actual is equal to expected
template <typename T, typename U>
inline void
check_equal(const T& actual,
            const U& expected,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::equal_to<>(), "==", msg, loc);
}

// Check if actual is not equal to expected
template <typename T, typename U>
inline void check_not_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::not_equal_to<>(), "!=", msg,
                          loc);
}

// Check if actual is greater than expected
template <typename T, typename U>
inline void check_greater(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::greater<>(), ">", msg, loc);
}

// Check if actual is less than expected
template <typename T, typename U>
inline void
check_less(const T& actual,
           const U& expected,
           std::string_view msg            = "",
           const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::less<>(), "<", msg, loc);
}

// Check if actual is greater than or equal to expected
template <typename T, typename U>
inline void check_greater_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::greater_equal<>(), ">=", msg,
                          loc);
}

// Check if actual is less than or equal to expected
template <typename T, typename U>
inline void check_less_equal(
    const T& actual,
    const U& expected,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    check_with_comparator(actual, expected, std::less_equal<>(), "<=", msg,
                          loc);
}

// Check a generic boolean condition
inline void
check(bool condition,
      std::string_view msg,
      const std::source_location& loc = std::source_location::current()) {
    if (!condition) {
        throw DetailedException(msg, loc);
    }
}

// Null pointer check
inline void check_not_null(
    const void* ptr,
    std::string_view msg            = "Pointer must not be null",
    const std::source_location& loc = std::source_location::current()) {
    if (ptr == nullptr) {
        throw DetailedException(msg, loc);
    }
}

// Check power of 2
template <std::integral Size>
inline void check_power_of_2(
    Size value,
    std::string_view msg            = "",
    const std::source_location& loc = std::source_location::current()) {
    if ((value & (value - 1)) != 0) {
        std::string composed =
            msg.empty()
                ? std::format("Check failed: {} is not a power of 2", value)
                : std::format("{} must be power of 2 (got {})", msg, value);
        throw DetailedException(composed, loc);
    }
}

// Check if value is even
template <std::unsigned_integral Size>
inline void
check_even(Size value,
           std::string_view msg            = "",
           const std::source_location& loc = std::source_location::current()) {
    if ((value & 1U) != 0U) {
        throw DetailedException(msg, loc);
    }
}

// Check if index is within range [0, size)
template <std::integral Index, std::integral Size>
inline void
check_range(Index index,
            Size size,
            std::string_view msg            = "",
            const std::source_location& loc = std::source_location::current()) {
    if (index < 0 || static_cast<std::make_unsigned_t<Index>>(index) >=
                         static_cast<std::make_unsigned_t<Size>>(size)) {
        std::string composed =
            msg.empty()
                ? std::format("Index {} out of range [0, {})", index, size)
                : std::format("{} (index {} >= size {})", msg, index, size);
        throw DetailedException(composed, loc);
    }
}

/**
 * @brief Signal that a branching step would exceed its output workspace.
 *
 * Thrown *before* any out-of-bounds write happens, so the caller can recover
 * (the batched prune drivers catch this and retry with a halved batch).
 * Deliberately a std::overflow_error (not DetailedException) so that the
 * recovery path can catch it without also swallowing unrelated failures.
 */
[[noreturn]] inline void throw_branch_overflow(std::string_view where,
                                               std::size_t n_leaves_branched,
                                               std::size_t capacity,
                                               std::size_t n_leaves) {
    throw std::overflow_error(
        std::format("{}: branching workspace overflow: n_leaves_branched={} > "
                    "capacity={} (n_leaves={} in input batch); "
                    "reduce batch or raise branch_max",
                    where, n_leaves_branched, capacity, n_leaves));
}

/**
 * @brief Pre-flight capacity check for a CPU branch step.
 *
 * Sums the per-leaf product of the per-dimension child counts and throws
 * before the write loop runs if the result would not fit.
 *
 * @param scratch_counts Per-leaf, per-dimension child counts.
 * @param params_stride  Stride between leaves inside @p scratch_counts.
 * @param n_dims         Number of dimensions that contribute to the product.
 * @param dim_offset     First contributing dimension within a leaf.
 * @param capacity       Number of output leaves that fit.
 */
template <typename CountT>
inline void check_branch_capacity(std::string_view where,
                                  const CountT* scratch_counts,
                                  std::size_t n_leaves,
                                  std::size_t params_stride,
                                  std::size_t n_dims,
                                  std::size_t dim_offset,
                                  std::size_t capacity) {
    std::size_t total = 0;
    for (std::size_t i = 0; i < n_leaves; ++i) {
        std::size_t prod = 1;
        for (std::size_t k = 0; k < n_dims; ++k) {
            prod *= static_cast<std::size_t>(
                scratch_counts[(i * params_stride) + dim_offset + k]);
        }
        total += prod;
    }
    if (total > capacity) {
        throw_branch_overflow(where, total, capacity, n_leaves);
    }
}

} // namespace loki::error_check
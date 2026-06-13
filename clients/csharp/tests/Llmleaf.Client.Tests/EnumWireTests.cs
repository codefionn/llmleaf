using System.Reflection;
using Xunit;

namespace Llmleaf.Client.Tests;

// The public enums must stay in lockstep with the generated Google.Protobuf enums (the schema
// proof): same members, same numeric values, same proto value names. This test reflects over the
// generated [OriginalName] attributes and asserts the public [WireName] attributes match — so a
// proto change that regenerates Llmleaf.cs but not the public enums fails here.
public sealed class EnumWireTests
{
    [Theory]
    [InlineData(typeof(Role), typeof(Llmleaf.V1.Role))]
    [InlineData(typeof(FinishReason), typeof(Llmleaf.V1.FinishReason))]
    [InlineData(typeof(BatchStatus), typeof(Llmleaf.V1.BatchStatus))]
    public void PublicEnumMatchesGeneratedProtoEnum(Type publicEnum, Type genEnum)
    {
        var publicNames = GetWireNames(publicEnum);
        var genNames = GetOriginalNames(genEnum);
        Assert.Equal(genNames, publicNames);
    }

    private static Dictionary<int, string> GetWireNames(Type enumType)
    {
        var map = new Dictionary<int, string>();
        foreach (var f in enumType.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
        {
            var value = (int)f.GetRawConstantValue()!;
            var attr = f.GetCustomAttribute<WireNameAttribute>();
            map[value] = attr?.ProtoName ?? f.Name;
        }
        return map;
    }

    private static Dictionary<int, string> GetOriginalNames(Type enumType)
    {
        var map = new Dictionary<int, string>();
        foreach (var f in enumType.GetFields(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static))
        {
            var value = (int)f.GetRawConstantValue()!;
            var attr = f.GetCustomAttribute<Google.Protobuf.Reflection.OriginalNameAttribute>();
            map[value] = attr?.Name ?? f.Name;
        }
        return map;
    }

    [Fact]
    public void WireTokensAreLowercasedProtoNames()
    {
        Assert.Equal("tool_calls", EnumWire.ToWire(FinishReason.ToolCalls));
        Assert.Equal("assistant", EnumWire.ToWire(Role.Assistant));
        Assert.Equal("in_progress", EnumWire.ToWire(BatchStatus.InProgress));
        // Unspecified zero value -> field absent (null).
        Assert.Null(EnumWire.ToWire(Role.Unspecified));
    }

    [Fact]
    public void FromWireIsCaseInsensitiveAndDefaultsToUnspecified()
    {
        Assert.Equal(FinishReason.ToolCalls, EnumWire.FromWire<FinishReason>("TOOL_CALLS"));
        Assert.Equal(Role.User, EnumWire.FromWire<Role>("user"));
        Assert.Equal(BatchStatus.Unspecified, EnumWire.FromWire<BatchStatus>("nonexistent"));
        Assert.Equal(Role.Unspecified, EnumWire.FromWire<Role>(null));
    }
}
